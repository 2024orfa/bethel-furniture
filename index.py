# import os
# from dotenv import load_dotenv

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone as PineconeClient, ServerlessSpec

# # Load environment variables
# load_dotenv()

# pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# # Read the extracted content from the text file
# file_path = "extracted_content.txt"

# with open(file_path, "r", encoding="utf-8") as file:
#     text_content = file.read()

# # Initialize the RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=10)

# # Split the text into chunks
# chunks = text_splitter.split_text(text_content)

# # Initialize the Sentence Transformer model
# model = SentenceTransformer('all-mpnet-base-v2')

# # Embed the chunks
# embeddings = model.encode(chunks)

# # Initialize Pinecone client
# client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)

# # Check if the index already exists, if not, create it
# if pinecone_index_name not in [index.name for index in client.list_indexes()]:
#     client.create_index(
#         pinecone_index_name,
#         dimension=768,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )

# # Connect to the created index
# index = client.Index(pinecone_index_name)

# # Prepare and index the data
# docs = []
# for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#     doc = {
#         'id': str(i),
#         'values': embedding.tolist(),
#         'metadata': {'text': chunk}
#     }
#     docs.append(doc)

# # Function to split documents into smaller batches
# def batch_documents(documents, batch_size):
#     for i in range(0, len(documents), batch_size):
#         yield documents[i:i + batch_size]

# # Define a batch size
# batch_size = 100

# # Upsert the documents to Pinecone in batches
# for batch in batch_documents(docs, batch_size):
#     try:
#         index.upsert(vectors=batch)
#     except Exception as e:
#         print(f"Error during upsert operation for batch: {e}")

# print(f"Successfully indexed {len(docs)} chunks into Pinecone")

import os
from dotenv import load_dotenv
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load environment variables
load_dotenv()

# Retrieve environment variables
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = openai_api_key

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
    # OpenAI allows up to 2048 tokens per request; adjust batch size accordingly
    batch_size = 100  # Adjust based on your needs and rate limits
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"  # You can choose other models if preferred
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
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
for batch in batch_documents(docs, batch_size):
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"Error during upsert operation for batch: {e}")
        # Optionally implement retry logic here

print(f"Successfully indexed {len(docs)} chunks into Pinecone")

