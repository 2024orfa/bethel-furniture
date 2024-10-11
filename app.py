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
- When asked about price, shipping, discounts, or contact, provide a very detailed and complete response.
- When asked about shipping costs, explain the following: 
    - There is a fixed shipping cost of 450 Rands for addresses within 50 km from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng.
    - For locations beyond 50 km, the cost is 15 Rands per additional kilometer, added to the base shipping cost.
    - You can calculate the approximate shipping cost for the user based on their distance from this address, but also mention that the price is approximate and they should contact the store for an exact amount.
- When asked about reviews, give a comprehensive overview based on the available information.
- When providing price information:
    - **Name of the item** should be in bold.
    - **Dimensions** of the item should be listed.
    - **Price** of the item should be stated.
    - **Old Price** of the item should be included if available.
    - **Weight** of the item should be included if available.
    - Any other relevant details should be provided.
- When asked about contact information, extract and present the contact details from the context in the following format:
    - **Phone:** [Phone Number(s)]
    - **Email:** [Email Address(es)]
    - **Address:** [Physical Address]
    - Ensure each contact method is on a separate line and properly formatted with bold labels.
    - If multiple phone numbers or email addresses are available, list them separated by commas.
    - If any contact method is missing from the context, omit that line from the response.
If the context does not contain the answer, indicate that the information is not available.

---

**Example 1: Price Inquiry**

Context: The Deluxe Wooden Table is crafted from premium oak wood, measuring 120cm in length, 60cm in width, and 75cm in height. It weighs approximately 30kg. The current price is 2,500 Rands, reduced from the old price of 3,500 Rands.

Question: What is the price of the Deluxe Wooden Table?

Answer:
**Deluxe Wooden Table**
- **Dimensions:** 120cm (L) x 60cm (W) x 75cm (H)
- **Weight:** 30kg
- **Current Price:** 2,500 Rands
- **Old Price:** 3,500 Rands
- **Additional Details:** Comes with a 2-year warranty and free assembly service.

---

**Example 2: Shipping Cost Inquiry**

Context: Our online store offers shipping services from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng. The fixed shipping cost is 450 Rands for addresses within 50 km. For locations beyond 50 km, the cost is 15 Rands per additional kilometer.

Question: How much would shipping cost to Cape Town, which is 750 km away?

Answer:
There is a fixed shipping cost of **450 Rands** for addresses within 50 km from **12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng**.
For locations beyond 50 km, the cost is **15 Rands per additional kilometer**.
Cape Town is **750 km** away, so the additional kilometers are **700 km** (750 km - 50 km).
- **Additional Shipping Cost:** 700 km * 15 Rands = **10,500 Rands**
- **Total Approximate Shipping Cost:** 450 Rands + 10,500 Rands = **10,950 Rands**
  
*Please note that this is an approximate price. For an exact amount, please contact our store.*

---

**Example 3: Contact Information Inquiry**

Context: You can contact us through several methods. Here are the details:
- **Phone:** You can call us at 060 123 4567 or 061 765 4321.
- **Email:** Reach us via email at support@example.com or sales@example.com.
- **Address:** Our headquarters are located at 123 Main Street, Springfield, 0001.

Question: How can I contact your company?

Answer:
- **Phone:** 060 123 4567, 061 765 4321
- **Email:** support@example.com, sales@example.com
- **Address:** 123 Main Street, Springfield, 0001

Feel free to reach out to us through any of these methods for assistance!

---

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

