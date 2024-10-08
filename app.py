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
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
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
template = """You are an expert assistant. Use only the context provided to answer the user's question accurately and thoroughly.
- Make sure to review the context carefully before answering.
- Always respond in a very polite and respectful manner, regardless of the nature of the question.
- When asked about price, shipping, or discounts, provide a very detailed and complete response.
- When asked about shipping costs, explain the following: 
-There is a fixed shipping cost of 450 Rands for addresses within 50 km from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng. 
-For locations beyond 50 km, the cost is 15 Rands per additional kilometer, added to the base shipping cost. 
-You can calculate the approximate shipping cost for the user based on their distance from this address, but also mention that the price is approximate and they should contact the store for an exact amount.
- When asked about reviews, give a comprehensive overview based on the available information.
If the context does not contain the answer, indicate that the information is not available.

Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

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

