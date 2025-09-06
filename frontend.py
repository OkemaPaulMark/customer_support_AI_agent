import streamlit as st
import requests
from typing import List, Dict

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/chat"

# Initialize chat history in Streamlit's session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("Customer Support Agent")
st.write(
    "Your autonomous AI assistant for customer support. Ask questions about our services and create support tickets."
)

# Display chat messages from history on app rerun
for message in st.session_state.conversation_history:
    with st.chat_message(message["type"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history immediately
    st.session_state.conversation_history.append({"type": "human", "content": prompt})

    # Create a placeholder for the agent's response (with a spinner)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # Prepare data for FastAPI request
            fastapi_request_data = {
                "user_input": prompt,
                "conversation_history": st.session_state.conversation_history,
            }

            agent_response_content = ""
            try:
                response = requests.post(FASTAPI_URL, json=fastapi_request_data)
                response.raise_for_status()  # Raise an exception for HTTP errors
                api_response = response.json()
                agent_response_content = api_response["agent_response"]
                updated_conversation_history = api_response[
                    "updated_conversation_history"
                ]

                # Update Streamlit's session state with the full history from the API
                st.session_state.conversation_history = updated_conversation_history

            except requests.exceptions.RequestException as e:
                st.error(
                    f"Could not connect to the agent backend. Please ensure it is running. Error: {e}"
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

            # Display the actual agent response in the same chat message container
            st.markdown(agent_response_content)
