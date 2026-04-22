import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Endee RAG Chatbot")
st.write("Setup is working.")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    st.success("OPENAI_API_KEY loaded successfully.")
else:
    st.warning("OPENAI_API_KEY not found in .env file.")