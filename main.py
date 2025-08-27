import os
from dotenv import load_dotenv
from agent_graph import process_graph_with_agent
from database_utils import check_database_connection, create_support_ticket
from rag_system import get_knowledge_base_stats, initialize_vectorstore
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)  # Import BaseMessage and its subclasses

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY", "your_langsmith_api_key"
)


# Conversation State Management
class ConversationState:
    def __init__(self):
        self.history: list[BaseMessage] = []

    def add_message(self, role: str, content: str):
        if role == "user":
            self.history.append(HumanMessage(content=content))
        elif role == "assistant":
            self.history.append(AIMessage(content=content))

    def get_history(self) -> list[BaseMessage]:
        return self.history[-10:]  # Keep last 10 messages


conversation_state = ConversationState()


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


# Main Function
def main():
    """Main function with autonomous agent."""
    from database_utils import init_ticket_db

    init_ticket_db()

    print("Autonomous Customer Support Agent Ready!")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    # Check database connection
    if check_database_connection():
        print("Database connected successfully")
    else:
        print("Database connection failed")

    # Initialize RAG system
    print("Initializing knowledge base...")
    initialize_vectorstore()
    rag_stats = get_knowledge_base_stats()
    print(f"Knowledge Base: {rag_stats['document_count']} documents\n")

    print(" - I can autonomously:")
    print("  - Search database for team info & FAQs")
    print("  - Search documentation using RAG")
    print("  - Create support tickets when needed")
    print("  - Check ticket status")
    print("  - Handle complex multi-step queries!\n")

    # Interaction Loop

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Thank you for using our support service.")
                break

            # Handle conversational queries
            if is_conversational(user_input) or is_goodbye(user_input):
                response = handle_conversational_query(user_input)
                print(f"Assistant: {response}\n")
                continue

            # Use agent for everything else
            print("Thinking...", end="", flush=True)
            response_message = process_graph_with_agent(
                user_input, conversation_state.get_history()
            )
            response_content = (
                response_message.content
                if hasattr(response_message, "content")
                else str(response_message)
            )
            print(f"\rAssistant: {response_content}\n")

            # Update conversation history
            conversation_state.add_message("user", user_input)
            conversation_state.add_message("assistant", response_content)

            # Handle unanswered queries

            if response_content.strip() == "___NO_INFO_FOUND___":
                print("Assistant: I apologize, I don't have an answer to that.")
                create_ticket = (
                    input(
                        "Would you like me to create a support ticket for this issue? (yes/no): "
                    )
                    .strip()
                    .lower()
                )
                if create_ticket in ["yes", "y"]:
                    ticket_id = create_support_ticket("anonymous", user_input)
                    print(
                        f"Support ticket #{ticket_id} created successfully. Our support team will reach out to you soon.\n"
                    )
                else:
                    print("Okay, no ticket created.\n")

        except KeyboardInterrupt:
            print("\nGoodbye! Thank you for using our support service.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break


# Run Main
if __name__ == "__main__":
    main()
