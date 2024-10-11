from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from pinecone import Pinecone as PineconeClient
from langchain.embeddings import OpenAIEmbeddings
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Retrieve environment variables
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Pinecone client and connect to the index
client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)
index = client.Index(pinecone_index_name)

# Initialize the vectorstore and retriever
vectorstore = Pinecone.from_existing_index(
    index_name=pinecone_index_name,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# Define the prompt template with conversation history
template = """You are an expert assistant. Use only the context provided to answer the user's question accurately and thoroughly.
- Make sure to review the context carefully before answering.
- Always respond in a very polite and respectful manner, regardless of the nature of the question.
- When asked about price, shipping, discounts, or contact, provide a very detailed and complete response.
- When asked about shipping costs, explain the following: 
    - There is a fixed shipping cost of 450 Rands for addresses within 50 km from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng.
    - For locations beyond 50 km, the cost is 15 Rands per additional kilometer, added to the base shipping cost.
    - You can calculate the approximate shipping cost for the user based on their distance from this address, but also mention that the price is approximate and they should contact the store for an exact amount.
- When asked about reviews, give a comprehensive overview based on the available information.
- When providing price information make sure to state old price, new price, dimensions, and weight of item if available.

If the context does not contain the answer, indicate that the information is not available.


**Conversation History:**
{conversation_history}

**Question:** {question}
Answer:"""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="conversation_history")

# Initialize the OpenAI model
# Note: Ensure the model name is correct. It was "gpt-4o-mini" in your original code, which might be a typo.
# If you intend to use GPT-4, use "gpt-4"
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")  # Changed from "gpt-4o-mini" to "gpt-4"

# Create the Conversation Chain
conversation_chain = ConversationChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    retriever=retriever
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    conversation_history = data.get('conversation_history', '')  # Entire conversation history

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Update memory with provided conversation history
    conversation_chain.memory.conversation_history = conversation_history

    # Invoke the conversation chain with user input
    try:
        answer = conversation_chain.run(user_input)
        # Retrieve updated conversation history
        updated_conversation_history = conversation_chain.memory.conversation_history
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'response': answer, 'conversation_history': updated_conversation_history})

if __name__ == '__main__':
    app.run()
