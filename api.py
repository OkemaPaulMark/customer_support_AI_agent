from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from agent_graph import process_graph_with_agent
from database_utils import create_support_ticket

app = FastAPI()


class Message(BaseModel):
    content: str
    type: str  # "human" or "ai"


class ChatRequest(BaseModel):
    user_input: str
    conversation_history: List[Message] = []


class ChatResponse(BaseModel):
    agent_response: str
    updated_conversation_history: List[Message]


def convert_to_langchain_messages(history: List[Message]) -> List[BaseMessage]:
    langchain_history = []
    for msg in history:
        if msg.type == "human":
            langchain_history.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            langchain_history.append(AIMessage(content=msg.content))
    return langchain_history


def convert_from_langchain_messages(history: List[BaseMessage]) -> List[Message]:
    generic_history = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            generic_history.append(Message(content=msg.content, type="human"))
        elif isinstance(msg, AIMessage):
            generic_history.append(Message(content=msg.content, type="ai"))
    return generic_history


# Conversational Query Functions
def is_conversational(query: str):
    """Check if the query is conversational."""
    conversational_keywords = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    query_lower = query.lower().strip()
    return any(keyword in query_lower for keyword in conversational_keywords)


def is_goodbye(query: str):
    """Check if the query is a goodbye."""
    goodbye_keywords = ["bye", "goodbye", "see you", "farewell", "take care"]
    query_lower = query.lower().strip()
    return any(keyword in query_lower for keyword in goodbye_keywords)


def handle_conversational_query(query: str):
    """Handle conversational queries."""
    query_lower = query.lower().strip()

    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "howdy"]):
        return "Hello! I'm your autonomous customer support agent. How can I help you today?"

    if "how are you" in query_lower:
        return "I'm functioning well, thank you! I'm here to help you with any questions using my available tools."

    if any(morning in query_lower for morning in ["good morning", "morning"]):
        return "Good morning! What can I assist you with today?"

    if is_goodbye(query_lower):
        return "Goodbye! Thank you for chatting with me. Have a great day!"

    return "I'm here to help! How can I assist you today?"


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        # Handle conversational queries first
        if is_conversational(request.user_input) or is_goodbye(request.user_input):
            response_content = handle_conversational_query(request.user_input)
            # Directly append to the incoming history and convert once
            temp_history = convert_to_langchain_messages(request.conversation_history)
            temp_history.append(HumanMessage(content=request.user_input))
            temp_history.append(AIMessage(content=response_content))
            updated_conversation_history = convert_from_langchain_messages(temp_history)
            return ChatResponse(
                agent_response=response_content,
                updated_conversation_history=updated_conversation_history,
            )

        langchain_history = convert_to_langchain_messages(request.conversation_history)

        # Add current user input to history for agent processing
        langchain_history.append(HumanMessage(content=request.user_input))

        response_history = process_graph_with_agent(
            request.user_input, langchain_history
        )

        # The last message in response_history is the agent's latest response
        agent_response_message = response_history[-1]
        agent_response_content = agent_response_message.content

        updated_conversation_history = convert_from_langchain_messages(response_history)

        if agent_response_content.strip() == "___NO_INFO_FOUND___":
            pass

        return ChatResponse(
            agent_response=agent_response_content,
            updated_conversation_history=updated_conversation_history,
        )
    except Exception as e:
        import traceback

        print("\n--- FastAPI Error Traceback ---")
        traceback.print_exc()  # Print full traceback to console
        print("-------------------------------")
        raise HTTPException(status_code=500, detail=str(e))
