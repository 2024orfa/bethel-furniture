import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import time  # For optional delay to simulate more visible progress

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Retrieve environment variables
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Read the extracted content from the text file
file_path = "extracted_content.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=10)

# Split the text into chunks
chunks = text_splitter.split_text(text_content)

# Function to get OpenAI embeddings
def get_openai_embeddings(text_list):
    embeddings = []
    batch_size = 5  # Define the batch size based on your needs
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            # Ensure correct usage of OpenAI client
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            # Assuming response.data is a list of embeddings
            batch_embeddings = [item.embedding for item in response.data]  # Use dot notation
            embeddings.extend(batch_embeddings)
            print(f"Processed batch {i // batch_size + 1} of {len(text_list) // batch_size + 1}")
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error for batch starting at index {i}: {e}")
            # Optionally implement retry logic here
    return embeddings

# Embed the chunks using OpenAI
embeddings = get_openai_embeddings(chunks)

# Initialize Pinecone client
client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)

# Check if the index already exists, if not, create it
if pinecone_index_name not in [index.name for index in client.list_indexes()]:
    client.create_index(
        pinecone_index_name,
        dimension=1536,  # The dimension for 'text-embedding-ada-002' is 1536
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the created index
index = client.Index(pinecone_index_name)

# Prepare and index the data
docs = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    doc = {
        'id': str(i),
        'values': embedding,  # OpenAI embeddings are already lists
        'metadata': {'text': chunk}
    }
    docs.append(doc)

# Function to split documents into smaller batches
def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Define a batch size
batch_size = 100  # Adjust based on Pinecone's upsert limits

# Upsert the documents to Pinecone in batches
for i, batch in enumerate(batch_documents(docs, batch_size), start=1):
    try:
        index.upsert(vectors=batch)
        print(f"Successfully upserted batch {i} of {len(docs) // batch_size + 1}")
        # Optional: Add delay to simulate more visible progress (remove in production)
        time.sleep(0.5)
    except Exception as e:
        print(f"Error during upsert operation for batch {i}: {e}")
        # Optionally implement retry logic here

print(f"Successfully indexed {len(docs)} chunks into Pinecone")
