

# %%
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read the extracted content from the text file
file_path = "extracted_content.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

# Initialize the RecursiveCharacterTextSplitter
# Adjust `chunk_size` and `chunk_overlap` based on your preference
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=10)

# Split the text into chunks
chunks = text_splitter.split_text(text_content)

# Print the chunks to verify the result
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n {chunk}\n")

from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Embed the filtered chunks
embeddings = model.encode(chunks)

# Print each embedding
for i, embedding in enumerate(embeddings):
    print(f"Embedding for Chunk {i+1}:\n{embedding}\n")


# %% [markdown]
# ## Create Index

# %%
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer


# Initialize Pinecone with PineconeClient
client = Pinecone(api_key="0990c15d-15bf-4dd8-80af-2361a2df1aa3", environment="us-east-1")
# Define the index name
index_name = "bethel"
# Check if the index already exists, if not, create it
if index_name not in client.list_indexes().names():
    client.create_index(
        index_name, 
        dimension=768, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )

# %%
# # Initialize Pinecone with PineconeClient
# client = Pinecone(api_key="0990c15d-15bf-4dd8-80af-2361a2df1aa3", environment="us-east-1")
# # Define the index name
# client.delete_index(index_name)

# %%
import numpy as np
# Connect to the created index
index = client.Index(index_name)

# Ensure the sentence transformer model is loaded for both indexing and searching
model = SentenceTransformer('all-mpnet-base-v2')

# Prepare and index the data
docs = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    doc = {
        'id': str(i),
        'values': embedding.tolist(),
        'metadata': {'text': chunk}
    }
    docs.append(doc)

# Function to split documents into smaller batches
def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Define a batch size
batch_size = 100

# Upsert the documents to Pinecone in batches
for batch in batch_documents(docs, batch_size):
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"Error during upsert operation for batch: {e}")

print(f"Successfully indexed {len(docs)} chunks into Pinecone")

# %%
# Initialize Pinecone vectorstore with the correct embedding function
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

# # Load environment variables from .env file
load_dotenv()

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

vectorstore = Pinecone.from_existing_index(
    index_name=pinecone_index_name,
    embedding=embeddings# Pass the embedding function, not embeddings directly
)
retriever=vectorstore.as_retriever()


template = """You are an expert assistant. Use the context provided to answer the user's question accurately.
Make sure you review throughly before answering
Always answer in a very polite manner no matter the question
Make sure to give very deatiled responses when asked a question about price,shipping or discounts
Make sure to give deatiled information when asked about reviews
If the context does not contain the answer, indicate that the information is not available.

Context: {context}
Question: {question}
Answer:"""
prompt= ChatPromptTemplate.from_template(template)


model = ChatOpenAI(openai_api_key=openai_api_key,
                   model="gpt-3.5-turbo")


chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
    
)


chain.invoke("How much does Black Modern Bed Pedestal and Headboard 45kgcost?")
#chain.invoke("what was the original price of Black Modern Bed Pedestal and Headboard")
#chain.invoke("what are the reviews in bethel furniture")
#chain.invoke("call us")





