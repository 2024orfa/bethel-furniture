from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone as PineconeClient
from langchain_community.embeddings import OpenAIEmbeddings
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Initialize Pinecone client and connect to index
client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)
index = client.Index(pinecone_index_name)

# Initialize the vectorstore and retriever
vectorstore = Pinecone.from_existing_index(
    index_name=pinecone_index_name,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# Define the prompt template
template = """You are an expert assistant. Use the context provided to answer the user's question accurately.
Make sure you review thoroughly before answering.
Always answer in a very polite manner no matter the question.
Make sure to give very detailed responses when asked a question about price, shipping, or discounts.
Make sure to give detailed information when asked about reviews.
If the context does not contain the answer, indicate that the information is not available.

Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Create the chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Invoke the chain with user input
    try:
        answer = chain.invoke(user_input)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run()

