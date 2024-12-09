# from flask import Flask, request, jsonify, render_template
# import os
# from dotenv import load_dotenv
# from langchain.vectorstores import Pinecone
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from pinecone import Pinecone as PineconeClient
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load environment variables
# load_dotenv()

# # Retrieve environment variables
# pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI embeddings
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Initialize Pinecone client and connect to the index
# client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_environment)
# index = client.Index(pinecone_index_name)

# # Initialize the vectorstore and retriever
# vectorstore = Pinecone.from_existing_index(
#     index_name=pinecone_index_name,
#     embedding=embeddings
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# # Define the prompt template with conversation history and additional formatting instructions
# # Updated prompt template
# template = """
# You are an expert assistant. Use only the context provided to answer the user's question accurately and thoroughly.
# - Always prioritize the context over the conversation history when answering.
# - Make sure to review the context carefully before answering.
# - Respond in a polite and respectful manner, regardless of the nature of the question.
# - When asked about price, shipping, discounts, or contact, provide a detailed and complete response.
# - Always use HTML formatting for the response (do not use Markdown, code blocks, backticks, or asterisks).
# - Use <b> for bold labels (e.g., <b>Address:</b>).
# - Use <br> for line breaks after each piece of information.
# - For lists, use <ul> and <li> tags.
# - Ensure the response is well-structured using HTML and do not include any code blocks or backticks.

# **Shipping Calculation Instructions:**
# - For shipping cost calculations, use the following formula:
#     - There is a flat rate of 450 Rands for addresses within 50 km from Bethel Furniture's address.
#     - For locations beyond 50 km, the additional cost is 15 Rands per additional kilometer.
#     - Example: For an order 100 km away, the additional distance is 50 km (100 km - 50 km), so the additional cost is 50 km * 15 Rands.

# **Formatting Instructions:**
# - Use <b> for bold labels (e.g., <b>Address:</b>).
# - Use <br> for line breaks after each piece of information.
# - For lists like multiple emails, use <ul> and <li> tags.
# - Ensure the response is well-structured using HTML and do not include any code blocks or backticks.

# **Examples:**

# Example 1: Shipping Cost Inquiry

# Context: Our online store offers shipping services from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng. The fixed shipping cost is 450 Rands for addresses within 50 km. For locations beyond 50 km, the cost is 15 Rands per additional kilometer.

# Question: How much would shipping cost to Cape Town, which is 750 km away?

# Answer:
# There is a fixed shipping cost of <b>450 Rands</b> for addresses within 50 km from <b>12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng</b>.<br>
# For locations beyond 50 km, the cost is <b>15 Rands per additional kilometer</b>.<br>
# Cape Town is <b>750 km</b> away, so the additional kilometers are <b>700 km</b> (750 km - 50 km).<br>
# - <b>Additional Shipping Cost:</b> 700 km * 15 Rands = <b>10,500 Rands</b><br>
# - <b>Total Approximate Shipping Cost:</b> 450 Rands + 10,500 Rands = <b>10,950 Rands</b>

# Example 2:

# <b>Address:</b> 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng<br>
# <b>Email:</b><br>
# <ul>
#   <li>bethel@bethelfurniture.com</li>
#   <li>info@bethelfurniture.com</li>
#   <li>bethel.ptyltd@gmail.com</li>
# </ul>
# <b>Phone Number:</b> 060 545 6671<br>
# <b>Website:</b> <a href="http://www.bethelfurniture.com">www.bethelfurniture.com</a>

# **Context (Priority):**
# {context}

# **Previous Conversation (Secondary):**
# {conversation_history}

# **Question:** {question}

# Answer:
# """

# # Create the prompt template
# prompt = ChatPromptTemplate.from_template(template)

# # Initialize the OpenAI model
# model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

# # Create the LLMChain
# chain = LLMChain(llm=model, prompt=prompt)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_input = data.get('message', '')
#     conversation_history = data.get('conversation_history', '')  # Expecting a string

#     if not user_input:
#         return jsonify({'error': 'No input provided'}), 400

#     try:
#         # Retrieve relevant documents based on user input
#         docs = retriever.get_relevant_documents(user_input)
#         context = "\n".join([doc.page_content for doc in docs])

#         # Debug: Print context to verify content
#         print(f"Retrieved context: {context}")

#         # Combine context and conversation history, with context prioritized
#         full_history = f"Context: {context}\n\nPrevious Conversation: {conversation_history}\n"

#         # Prepare prompt inputs
#         prompt_inputs = {
#             "context": context,
#             "conversation_history": conversation_history,
#             "question": user_input
#         }

#         # Get the response from the chain
#         answer = chain.run(prompt_inputs)
        
#         answer = answer.replace('```html', '').replace('```', '').strip()

#         # Limit conversation history to the last 10 interactions
#         history_lines = conversation_history.split("\n")
#         if len(history_lines) > 10:
#             conversation_history = "\n".join(history_lines[-10:])

#         # Update conversation history
#         updated_conversation_history = f"{full_history}User: {user_input}\nAssistant: {answer}\n"

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

#     return jsonify({'response': answer, 'conversation_history': updated_conversation_history})

# if __name__ == '__main__':
#     app.run(port=8000)

from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone as PineconeClient
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Retrieve environment variables
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")
whatsapp_token = os.getenv("WHATSAPP_TOKEN")  # Your WhatsApp API token
verify_token = os.getenv("VERIFY_TOKEN")  # Your verification token

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
- Always use HTML formatting for the response (do not use Markdown, code blocks, backticks, or asterisks).
- Use <b> for bold labels (e.g., <b>Address:</b>).
- Use <br> for line breaks after each piece of information.
- For lists, use <ul> and <li> tags.
- Ensure the response is well-structured using HTML and do not include any code blocks or backticks.

**Shipping Calculation Instructions:**
- For shipping cost calculations, use the following formula:
    - There is a flat rate of 450 Rands for addresses within 50 km from Bethel Furniture's address.
    - For locations beyond 50 km, the additional cost is 15 Rands per additional kilometer.
    - Example: For an order 100 km away, the additional distance is 50 km (100 km - 50 km), so the additional cost is 50 km * 15 Rands.

**Formatting Instructions:**
- Use <b> for bold labels (e.g., <b>Address:</b>).
- Use <br> for line breaks after each piece of information.
- For lists like multiple emails, use <ul> and <li> tags.
- Ensure the response is well-structured using HTML and do not include any code blocks or backticks.

**Examples:**

Example 1: Shipping Cost Inquiry

Context: Our online store offers shipping services from 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng. The fixed shipping cost is 450 Rands for addresses within 50 km. For locations beyond 50 km, the cost is 15 Rands per additional kilometer.

Question: How much would shipping cost to Cape Town, which is 750 km away?

Answer:
There is a fixed shipping cost of <b>450 Rands</b> for addresses within 50 km from <b>12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng</b>.<br>
For locations beyond 50 km, the cost is <b>15 Rands per additional kilometer</b>.<br>
Cape Town is <b>750 km</b> away, so the additional kilometers are <b>700 km</b> (750 km - 50 km).<br>
- <b>Additional Shipping Cost:</b> 700 km * 15 Rands = <b>10,500 Rands</b><br>
- <b>Total Approximate Shipping Cost:</b> 450 Rands + 10,500 Rands = <b>10,950 Rands</b>

Example 2:

<b>Address:</b> 12 Observatory Avenue, Observatory, Johannesburg, 2198 Gauteng<br>
<b>Email:</b><br>
<ul>
  <li>bethel@bethelfurniture.com</li>
  <li>info@bethelfurniture.com</li>
  <li>bethel.ptyltd@gmail.com</li>
</ul>
<b>Phone Number:</b> 060 545 6671<br>
<b>Website:</b> <a href="http://www.bethelfurniture.com">www.bethelfurniture.com</a>

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
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

# Create the LLMChain
chain = LLMChain(llm=model, prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
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

        answer = answer.replace('```html', '').replace('```', '').strip()

        # Limit conversation history to the last 10 interactions
        history_lines = conversation_history.split("\n")
        if len(history_lines) > 10:
            conversation_history = "\n".join(history_lines[-10:])

        # Update conversation history
        updated_conversation_history = f"{full_history}User: {user_input}\nAssistant: {answer}\n"

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'response': answer, 'conversation_history': updated_conversation_history})

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    data = request.get_json()
    print(f"Incoming Webhook Data: {data}")

    if "messages" in data["entry"][0]["changes"][0]["value"]:
        messages = data["entry"][0]["changes"][0]["value"]["messages"]
        for message in messages:
            sender_id = message["from"]
            user_message = message["text"]["body"]

            try:
                # Retrieve relevant documents
                docs = retriever.get_relevant_documents(user_message)
                context = "\n".join([doc.page_content for doc in docs])

                # Prepare inputs for chatbot logic
                prompt_inputs = {
                    "context": context,
                    "conversation_history": "",
                    "question": user_message
                }

                # Get chatbot response
                bot_response = chain.run(prompt_inputs)

                # Convert HTML to WhatsApp Markdown format
                formatted_response = convert_html_to_markdown(bot_response)

                # Send the response to the WhatsApp user
                send_whatsapp_message(sender_id, formatted_response)

            except Exception as e:
                print(f"Error processing message: {e}")

    return 'EVENT_RECEIVED', 200


def convert_html_to_markdown(html_response):
    """
    Converts HTML-formatted response to WhatsApp-compatible Markdown.
    """
    import re
    # Convert <b>...</b> to *...* for bold
    response = re.sub(r"<b>(.*?)</b>", r"*\1*", html_response)
    # Replace <br> with newline
    response = response.replace("<br>", "\n")
    # Replace <ul> and <li> with bullet points
    response = re.sub(r"<ul>\s*(<li>.*?</li>)\s*</ul>", lambda m: re.sub(r"<li>(.*?)</li>", r"- \1", m.group(1)), response)
    # Remove any remaining HTML tags
    response = re.sub(r"<.*?>", "", response)
    return response.strip()


def send_whatsapp_message(recipient_id, message):
    """
    Sends a message to a WhatsApp user using the WhatsApp Cloud API.
    """
    url = f"https://graph.facebook.com/v17.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")


# Example conversion function test
if __name__ == "__main__":
    test_html = """
    <b>Shipping Costs:</b><br>
    Delivery fees depend on the size of your order and the delivery destination.<br>
    Shipping from our address at <b>12 Observatory Avenue</b> has the following structure:<br>
    <ul>
        <li><b>Fixed Shipping Cost:</b> 450 Rands for addresses within 50 km.</li>
        <li>For locations beyond 50 km: An additional cost of 15 Rands per additional kilometer.</li>
    </ul>
    """
    print(convert_html_to_markdown(test_html))