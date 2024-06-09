import streamlit as st
import requests
import pandas as pd
import openai
import inspect
from openai import AzureOpenAI

st.set_page_config(layout="wide")


aoai_endpoint = st.secrets["AOAIEndpoint"]
aoai_api_key = st.secrets["AOAIKey"]  
deployment_name = st.secrets["AOAIDeploymentName"]

def create_chat_completion(deployment_name, messages):
    client = AzureOpenAI(
    azure_endpoint=aoai_endpoint,
    api_key=aoai_api_key,
    api_version="2024-02-01",
    )

    return client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
        stream=True,
    )

def handle_chat_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = st.write_stream(create_chat_completion(deployment_name, st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    st.write(
    """
    # Chat with Data

    This Streamlit dashboard is intended to show off capabilities of LLMs.
    """
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Await a user message and handle the chat prompt when it comes in.
    if prompt := st.chat_input("Enter a message:"):
        handle_chat_prompt(prompt)

if __name__ == "__main__":
    main()
