import streamlit as st
import os
from streamlit_chat import message
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from azure.core.credentials import AzureKeyCredential
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from streamlit_feedback import streamlit_feedback
from streamlit_chat import message as streamlit_message
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.docstore.document import Document
from azure.search.documents import SearchClient
from openai import OpenAI
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import shelve
from streamlit_feedback import streamlit_feedback
import json
import datetime
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
from langchain_community.chat_models import AzureChatOpenAI

#Using dotenv in order to access my secret keys, could be to the LLM, vector database, search etc. It keeps it connected to Azure
dotenv_path = 'C:/Users/U422967/OneDrive - Danfoss/Desktop/Master thesis - chatbot/.env'
load_dotenv(dotenv_path=dotenv_path)



#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), #Here is ChatGPT 3.5 turbo used
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),  # This is a common parameter name for API keys
    openai_api_version="2024-02-15-preview",

    
    temperature=0.1, #Controls randomness. Lowering the temperature means 
    #that the model will produce more repetitive and deterministic responses. 
    #Increasing the temperature will result in more unexpected or creative responses.
    
    max_tokens=800, #Set a limit on the number of tokens per model response. The API supports a maximum of 
    #MaxTokensPlaceholderDoNotTranslate tokens shared between the prompt (including system message, examples, message history, and user query) and 
    #the model's response. One token is roughly 4 characters for typical English text.
   
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")


#content_key="content": This parameter specifies the key or field in the Azure Cognitive Search index that 
#contains the main content to be searched and retrieved. In this context, 

#"content" refers to the field within the search index where the searchable text or data is stored. 
#This could be the body of a document, a description, or any other textual data that the retriever should focus 
#on when performing searches.

#top_k=10: This parameter determines the number of search results to retrieve. 
#Setting top_k to 10 means that the retriever will fetch the top 10 most relevant results based 
#on the query it processes. This is useful for limiting the amount of data returned and focusing on 
#the most relevant information.

retriever = AzureCognitiveSearchRetriever(
    content_key="content",
    top_k=10,
)

#Explore bullet points instead of long text
def load_chain():
    prompt_template = """
    - You are a helpful assistant specialized in internal strategies at Danfoss, with a focus on Environmental, Social, and Governance (ESG) aspects.
    - Your responses should directly address the query without restating the question. Aim to provide concise, relevant information drawn from Danfoss's annual reports, sustainability reports, ESG glossaries, and frequently asked questions.
    - Avoid verbose explanations and ensure your answers are straightforward and to the point.
    - Incorporate knowledge from relevant regulations such as Regulation (EU) No 537/2014, Directive 2004/109/EC, Directive 2006/43/EC, and Directive 2013/34/EU regarding corporate sustainability reporting when applicable to the questions.
    - Strive to include key insights and takeaways that are directly related to the inquiries, ensuring your responses are informative and factually accurate. 

    {context}

    Question: {question}
    Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # Use a custom retrieval function that includes source information
    retriever = AzureCognitiveSearchRetriever(content_key="content",top_k=10)

     # Here, instead of directly using AzureCognitiveSearchRetriever,
    # we'll use a custom function or adapt the retriever to include source information.
    #retriever = custom_retrieval_function

    # Construct the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=client,
        memory=memory,
        retriever=retriever,  # Note: This assumes your chain can use a custom retriever function
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return chain

chain = load_chain()

# Define a function to display chat messages and handle the feedback form
def display_and_capture_feedback(user_question, bot_response, i):
    # Display chat messages
    message(user_question, is_user=True, key=f"user_question_{i}")
    message(bot_response, is_user=False, key=f"bot_response_{i}")

    # Feedback form
    with st.form(key=f"feedback_form_{i}"):
        feedback_value = st.radio("Feedback", options=["Thumbs Up 👍", "Thumbs Down 👎"], key=f"feedback_{i}")
        feedback_text = st.text_area("Feedback Text", key=f"feedback_text_{i}")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_simplified = "up" if feedback_value == "Thumbs Up 👍" else "down"
            handle_feedback_submission(feedback_id=i, user_question=user_question, bot_response=bot_response, feedback_value=feedback_simplified, feedback_text=feedback_text)
            st.session_state.feedback_submitted = True  # Set the flag to True to indicate feedback was submitted

def save_chat_history():
    """Saves the current chat history to a JSON file with a timestamp."""
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=4)
    st.success(f"Chat saved to {filename}")

def clear_chat_history():
    """Clears the current chat history from the session state to start a new chat."""
    st.session_state.chat_history = []
    st.session_state.conversation_id = 0
    st.experimental_rerun()

#def handle_feedback(feedback_id, user_question, bot_response, feedback_value, feedback_text, filename="feedback.json"):
def handle_feedback_submission(feedback_id, user_question, bot_response, feedback_value, feedback_text, filename="feedback.json"):
    # Load existing data or initialize as an empty list
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)
            # Check if existing_data is not a list, initialize as a list
            if not isinstance(existing_data, list):
                existing_data = []
    else:
        existing_data = []

    # Create a new feedback entry
    feedback_entry = {
        "user_question": user_question,
        "bot_response": bot_response,
        "feedback_value": feedback_value,  # Directly use the feedback value; ensure it's "up" or "down"
        "feedback_text": feedback_text
    }

    # Append new feedback entry to the list
    existing_data.append(feedback_entry)

    # Save updated data back to the file
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)


# Add a flag to track whether feedback was submitted
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


#Layout for the interface:

# Revised main interface using st.chat_input and st.chat_message
st.title("Danfoss Chatbot demo")

logo_url = "danfosslogo.png"  # Change this to the path or URL of the Danfoss logo
st.image(logo_url, width=200)  # You can adjust the width to fit your UI design

#Using Streamlit's session_state feature to manage state across reruns of a Streamlit app. 
#Streamlit apps are re-executed from top to bottom each time an interaction occurs (like a button press or text input). 
#This design means that maintaining state (like variable values) across these reruns requires a special mechanism, 
#which is what session_state provides.

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

#def get_text():
#    input_text = st.text_input("How can I help?", "", key="input")
#    return input_text

#if user_input:
#    output = chain.run(question=user_input)
#    st.session_state.past.append(user_input)
#    st.session_state.generated.append(output)

# Define the initial setup for the session state if not already defined
if "last_input_id" not in st.session_state:
    st.session_state.last_input_id = 0
    st.session_state.generated = []
    st.session_state.past = []

# This function processes the user's input
def process_input(user_input):
    output = chain.run(question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    # Increment the id to ensure the next input widget has a unique key
    st.session_state.last_input_id += 1  

# Display the text input and associate it with the current last_input_id
user_input_key = f"input_{st.session_state.last_input_id}"
user_input = st.text_input("How can I help?", key=user_input_key)

# Process the input when it's submitted
if user_input:
    process_input(user_input)

# Display loop for messages and feedback
if "generated" in st.session_state:
    for i in range(len(st.session_state["generated"])):
        user_question = st.session_state["past"][i]
        bot_response = st.session_state["generated"][i]
        user_key = f"user_question_{i}"
        bot_key = f"bot_response_{i}"

        # Display user question and bot response
        message(user_question, is_user=True, key=user_key)
        message(bot_response, is_user=False, key=bot_key)

        # Only display the feedback form for the latest message
        if i == len(st.session_state["generated"]) - 1:
            with st.form(key=f"feedback_form_{i}"):
                feedback_value = st.radio("Feedback", options=["Thumbs Up 👍", "Thumbs Down 👎"], key=f"feedback_{i}")
                feedback_text = st.text_area("Feedback Text (optional) ", key=f"feedback_text_{i}")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    feedback_simplified = "up" if feedback_value == "Thumbs Up 👍" else "down"
                    handle_feedback_submission(feedback_id=i, user_question=user_question, bot_response=bot_response, feedback_value=feedback_simplified, feedback_text=feedback_text)

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
