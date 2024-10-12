from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone as PineconeClient
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Define the prompt template with conversation history and additional formatting instructions
template = """
You are an expert assistant. Use only the context provided to answer the user's question accurately and thoroughly.
- Always prioritize the context over the conversation history when answering.
- Make sure to review the context carefully before answering.
- Respond in a polite and respectful manner, regardless of the nature of the question.
- When asked about price, shipping, discounts, or contact, provide a detailed and complete response.

**Formatting Instructions:**
- For contact information requests, format each contact method (Email, Phone, Location, Website) on separate lines.
- Use bullet points for clarity.
- Bold the labels for each contact method.

**Examples:**

**Example 1: Shipping Cost Inquiry**

Context: Our online store offers shipping services from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng. The fixed shipping cost is 450 Rands for addresses within 50 km. For locations beyond 50 km, the cost is 15 Rands per additional kilometer.

Question: How much would shipping cost to Cape Town, which is 750 km away?

Answer:
There is a fixed shipping cost of **450 Rands** for addresses within 50 km from **12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng**.
For locations beyond 50 km, the cost is **15 Rands per additional kilometer**.
Cape Town is **750 km** away, so the additional kilometers are **700 km** (750 km - 50 km).
- **Additional Shipping Cost:** 700 km * 15 Rands = **10,500 Rands**
- **Total Approximate Shipping Cost:** 450 Rands + 10,500 Rands = **10,950 Rands**

*Please note that this is an approximate price. For an exact amount, please contact our store.*

**Context (Priority):**
{context}

**Previous Conversation (Secondary):**
{conversation_history}

**Question:** {question}
Answer:
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")

# Create the LLMChain
chain = LLMChain(llm=model, prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    conversation_history = data.get('conversation_history', '')  # Expecting a string

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        # Retrieve relevant documents based on user input
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])

        # Debug: Print context to verify content
        print(f"Retrieved context: {context}")

        # Combine context and conversation history, with context prioritized
        full_history = f"Context: {context}\n\nPrevious Conversation: {conversation_history}\n"

        # Prepare prompt inputs
        prompt_inputs = {
            "context": context,
            "conversation_history": conversation_history,
            "question": user_input
        }

        # Get the response from the chain
        answer = chain.run(prompt_inputs)

        # Limit conversation history to the last 10 interactions
        history_lines = conversation_history.split("\n")
        if len(history_lines) > 10:
            conversation_history = "\n".join(history_lines[-10:])

        # Update conversation history
        updated_conversation_history = f"{full_history}User: {user_input}\nAssistant: {answer}\n"

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'response': answer, 'conversation_history': updated_conversation_history})

if __name__ == '__main__':
    app.run(port=8000)
