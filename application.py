import streamlit as st
import os
import json
from streamlit_chat import message
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shelve
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
from langchain_community.chat_models import AzureChatOpenAI

# Load environment variables
dotenv_path = '/.env'
load_dotenv(dotenv_path=dotenv_path)

# Sidebar for adjusting LLM parameters
st.sidebar.title("LLM Parameters")
temperature = st.sidebar.slider(
    "Temperature (Higher values mean the model will take more risks)", 
    min_value=0.0, max_value=1.0, value=0.1, step=0.05
)
top_p = st.sidebar.slider(
    "Top_p (Controls the diversity of the output)", 
    min_value=0.0, max_value=1.0, value=1.0, step=0.05
)

# Store the LLM parameters in session state
if "temperature" not in st.session_state:
    st.session_state.temperature = temperature
if "top_p" not in st.session_state:
    st.session_state.top_p = top_p

# Add descriptions
st.sidebar.markdown("""
**Temperature**: A higher value means the model will take more risks, producing more diverse outputs. Lower values result in more deterministic and focused responses.
""")
st.sidebar.markdown("""
**Top_p**: Controls the diversity of the output by limiting the model to consider only the top p probability mass. Lower values make the output more focused and deterministic.
""")

# Initialize the Azure OpenAI client
client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    openai_api_version="2024-02-15-preview",
    temperature=temperature,
    top_p=top_p,
    max_tokens=800,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Define the retriever
retriever = AzureAISearchRetriever(
    content_key="metadata", 
    top_k=10, 
    index_name="langchain-vector-esg"
)

def load_chain():
    prompt_template = """
    #You are a specialized assistant at Danfoss focusing on internal strategies related to Environmental, Social, and Governance (ESG) aspects. Follow these guidelines for every response:

    #1. **Direct and Concise**: Answer the query directly without restating the question. Provide concise and to-the-point information.
    #2. **Source-Driven**: Base your answers on Danfoss's annual reports, sustainability reports, ESG glossaries, and frequently asked questions. Always reference specific documents when possible.
    #3. **Relevant Regulations**: Incorporate knowledge from the following regulations when applicable:
    #   - Regulation (EU) No 537/2014
    #   - Directive 2004/109/EC
    #   - Directive 2006/43/EC
    #   - Directive 2013/34/EU
    #4. **Insightful and Accurate**: Provide key insights and takeaways that are directly related to the inquiry. Ensure responses are factually accurate and avoid unnecessary verbosity.
    

    {context}

    Question: {question}
    Answer:
    """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=client,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Ensure the chain returns source documents with metadata
    )
    return chain

chain = load_chain()

def display_and_capture_feedback(user_question, bot_response, metadata, i):
    message(user_question, is_user=True, key=f"user_question_{i}")
    message(bot_response, is_user=False, key=f"bot_response_{i}")
    if "I'm sorry" not in bot_response and "please provide more information" not in bot_response and "I don't know" not in bot_response:
        st.markdown(f"**Source**: {metadata}", unsafe_allow_html=True)
    with st.form(key=f"feedback_form_{i}"):
        feedback_value = st.radio("Feedback", options=["Thumbs Up 👍", "Thumbs Down 👎"], key=f"feedback_{i}")
        feedback_text = st.text_area("Feedback Text", key=f"feedback_text_{i}")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            feedback_simplified = "up" if feedback_value == "Thumbs Up 👍" else "down"
            handle_feedback_submission(feedback_id=i, user_question=user_question, bot_response=bot_response, feedback_value=feedback_simplified, feedback_text=feedback_text)
            st.session_state.feedback_submitted = True

def handle_feedback_submission(feedback_id, user_question, bot_response, feedback_value, feedback_text, filename="feedback.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = []
    else:
        existing_data = []

    # Add user details to feedback
    user_details = {
        "full_name": st.session_state.full_name,
        "role_at_danfoss": st.session_state.role_at_danfoss
    }

    feedback_entry = {
        "user_question": user_question,
        "bot_response": bot_response,
        "feedback_value": feedback_value,
        "feedback_text": feedback_text,
        "user_details": user_details,
        "model_parameters": {
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p
        }
    }
    existing_data.append(feedback_entry)
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Custom CSS for improved visuals
st.markdown("""
    <style>
    .chat-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #FF0000;
    }
    .chat-instructions {
        font-size: 1.1em;
        color: #333;
        margin-bottom: 20px.
    }
    .feedback-form {
        margin-top: 10px.
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
        padding: 20px.
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for additional options
st.sidebar.title("Options")
if st.sidebar.button("Save Chat History"):
    save_chat_history(st.session_state.messages)
    st.sidebar.success("Chat history saved.")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.generated = []
    st.session_state.past = []
    st.sidebar.success("Chat history cleared.")

# Main UI
st.title("Danfoss Chatbot Demo")
st.image("danfosslogo.png", width=200)

st.markdown("<div class='chat-header'>Welcome to Danfoss Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-instructions'>This chatbot is trained on internal ESG and Sustainability documents, the 2023 Danfoss Annual report and the EU regulation regarding Corporate Sustainability reporting from 2022. It may not be able to answer questions outside this scope, and be aware that it can't retrieve specific documents, but are more of an AI Text generation/answering Chatbot. Always double check the facts before using the generated output for external use. </div>", unsafe_allow_html=True)

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "last_input_id" not in st.session_state:
    st.session_state.last_input_id = 0
    st.session_state.generated = []
    st.session_state.past = []

# User Details form
if "full_name" not in st.session_state or "role_at_danfoss" not in st.session_state:
    with st.form(key="user_details_form"):
        full_name = st.text_input("Full Name")
        role_at_danfoss = st.text_input("Role at Danfoss")
        submit_details = st.form_submit_button("Submit")
        if submit_details:
            st.session_state.full_name = full_name
            st.session_state.role_at_danfoss = role_at_danfoss

# Ensure user details are collected before proceeding
if "full_name" in st.session_state and "role_at_danfoss" in st.session_state:
    def process_input(user_input):
        with st.spinner("Processing..."):
            response = chain({"question": user_input})

            # Extract the answer and source documents from the response
            output = response.get('answer', 'No answer provided')
            source_documents = response.get('source_documents', [])

            # Extract the PDF file source from the metadata
            pdf_source = "No source available"
            if source_documents and "I'm sorry" not in output and "please provide more information" not in output and "I don't know" not in output:
                first_doc_metadata = source_documents[0].page_content
                try:
                    metadata_dict = json.loads(first_doc_metadata)
                    full_path = metadata_dict.get('source', 'No source available')
                    pdf_source = os.path.basename(full_path)
                except json.JSONDecodeError:
                    pdf_source = 'Metadata not in expected format'

        st.session_state.past.append(user_input)
        st.session_state.generated.append((output, pdf_source))
        st.session_state.last_input_id += 1

    user_input_key = f"input_{st.session_state.last_input_id}"
    user_input = st.text_input("How can I help?", key=user_input_key)

    if user_input:
        process_input(user_input)

    if "generated" in st.session_state:
        for i in range(len(st.session_state["generated"])):
            user_question = st.session_state["past"][i]
            bot_response, pdf_source = st.session_state["generated"][i]
            user_key = f"user_question_{i}"
            bot_key = f"bot_response_{i}"

            message(user_question, is_user=True, key=user_key)
            message(bot_response, is_user=False, key=bot_key)
            if "I'm sorry" not in bot_response and "please provide more information" not in bot_response and "I don't know" not in bot_response:
                st.markdown(f"**Source**: {pdf_source}", unsafe_allow_html=True)

            if i == len(st.session_state["generated"]) - 1:
                with st.form(key=f"feedback_form_{i}", clear_on_submit=True):
                    feedback_value = st.radio("Feedback", options=["Thumbs Up 👍", "Thumbs Down 👎"], key=f"feedback_{i}", horizontal=True)
                    feedback_text = st.text_area("Feedback Text (optional)", key=f"feedback_text_{i}")
                    submit_feedback = st.form_submit_button("Submit Feedback")
                    if submit_feedback:
                        feedback_simplified = "up" if feedback_value == "Thumbs Up 👍" else "down"
                        handle_feedback_submission(feedback_id=i, user_question=user_question, bot_response=bot_response, feedback_value=feedback_simplified, feedback_text=feedback_text)
                        st.session_state.feedback_submitted = True

    save_chat_history(st.session_state.messages)
else:
    st.warning("Please fill out your details to proceed.")





