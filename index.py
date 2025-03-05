import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import time
import docx
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Retrieve environment variables
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Constants (consider moving to .env or config file)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150
OPENAI_BATCH_SIZE = 5
PINECONE_BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-ada-002"  # Using the constant
RETRY_DELAY = 5  # seconds
MAX_RETRIES = 3

# Function to extract text from .docx file
def extract_text_from_docx(docx_file_path):
    try:
        doc = docx.Document(docx_file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logging.error(f"Error extracting text from {docx_file_path}: {e}")
        return ""

# Read the extracted content from the text file (existing code)
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return ""

# Function to get OpenAI embeddings with retry logic
def get_openai_embeddings(text_list):
    embeddings = []
    for i in range(0, len(text_list), OPENAI_BATCH_SIZE):
        batch = text_list[i:i + OPENAI_BATCH_SIZE]
        for attempt in range(MAX_RETRIES):  # Retry loop
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=EMBEDDING_MODEL  # Use the constant here
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logging.info(f"Processed batch {i // OPENAI_BATCH_SIZE + 1} of {len(text_list) // OPENAI_BATCH_SIZE + 1}")
                break  # Exit retry loop if successful
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAI API error for batch starting at index {i}, attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    logging.error(f"Failed to get embeddings after {MAX_RETRIES} attempts for batch starting at index {i}")
                    # Handle the failure - perhaps return partial results or skip the batch
                    return []  # Return an empty list if all retries fail
    return embeddings

# Initialize Pinecone client
pinecone = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)

# Explicitly lowercase the index name
if pinecone_index_name:
    pinecone_index_name = pinecone_index_name.lower()  # Convert to lowercase
else:
    logging.error("PINECONE_INDEX_NAME environment variable is not set.")
    exit()  # Or handle the missing variable as appropriate

# Check if the index already exists, if not, create it
try:
    if pinecone_index_name not in pinecone.list_indexes():
        logging.info(f"Creating Pinecone index: {pinecone_index_name}")
        pinecone.create_index(
            pinecone_index_name,
            dimension=1536,  # The dimension for 'text-embedding-ada-002' is 1536
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logging.info(f"Pinecone index {pinecone_index_name} created successfully.")
    else:
        logging.info(f"Pinecone index {pinecone_index_name} already exists.")
except Exception as e:
    logging.error(f"Error creating or checking Pinecone index: {e}")
    exit()  # Exit the program if index creation fails

# Connect to the created index
index = pinecone.Index(pinecone_index_name)

# Function to split documents into smaller batches for Pinecone upsert
def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


def process_data(text_content, docx_file_path):
    """
    Processes both .txt and .docx data sources, chunks, embeds, and upserts to Pinecone.
    """

    # Read and process the .docx file
    docx_text_content = extract_text_from_docx(docx_file_path)

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Combine the content of both files
    combined_text_content = text_content + "\n" + docx_text_content  # Combine .txt and .docx content

    # Split the combined content into chunks
    chunks = text_splitter.split_text(combined_text_content)

    # Embed the chunks using OpenAI
    embeddings = get_openai_embeddings(chunks)

    # Prepare and index the data
    docs = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            'id': str(i),
            'values': embedding,  # OpenAI embeddings are already lists
            'metadata': {'text': chunk}
        }
        docs.append(doc)

    # Upsert the documents to Pinecone in batches
    for i, batch in enumerate(batch_documents(docs, PINECONE_BATCH_SIZE), start=1):
        try:
            index.upsert(vectors=batch)
            logging.info(f"Successfully upserted batch {i} of {len(docs) // PINECONE_BATCH_SIZE + 1}")
            # Optional: Add delay to simulate more visible progress (remove in production)
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error during upsert operation for batch {i}: {e}")
            # Optionally implement retry logic here

    logging.info(f"Successfully indexed {len(docs)} chunks into Pinecone")

# Main execution
if __name__ == "__main__":
    # File paths
    file_path = "extracted_content.txt"
    docx_file_path = "Bethel Furniture FAQ.docx"  # Replace with the actual path to your .docx file

    # Read text content
    text_content = read_text_file(file_path)

    # Process data
    process_data(text_content, docx_file_path)