from langchain.tools import tool
from database_utils import database_query_tool as db_query, create_support_ticket, get_ticket_by_id, query_ticket_answer
from rag_system import rag_search

# Database Query Tool
@tool
def query_database_tool(question: str) -> str:
    """
    Search the database for information about team members, FAQs, past tickets, or other structured data.
    
    Returns the answer if found. If no information is available, returns a special marker
    '___NO_INFO_FOUND___' so the main program can prompt the user to create a support ticket.
    
    Args:
        question (str): The user's query to search in the database.
    
    Returns:
        str: The response from the database or '___NO_INFO_FOUND___' if no data is found.
    """
    try:
        # Check past resolved tickets first
        ticket_answer = query_ticket_answer(question)
        if ticket_answer:
            return ticket_answer

        # Normal database query
        result = db_query(question)
        if result and result.get("found"):
            return result["response"]
        else:
            # Special marker for main.py to detect missing info
            return "___NO_INFO_FOUND___"
    except Exception as e:
        return f"Database query error: {str(e)}"


# RAG Knowledge Base Tool
@tool
def query_rag_tool(question: str) -> str:
    """
    Search documentation and knowledge base using RAG system.
    """
    try:
        result = rag_search(question)
        return result if result else "No relevant information found in documentation."
    except Exception as e:
        return f"RAG search error: {str(e)}"


# Support Ticket Tool
@tool
def create_support_ticket_tool(user_question: str, user_name: str = "anonymous") -> str:
    """
    Ask the user if they want a support ticket created. 
    Store the actual user question as the issue.
    """
    # Ask user for confirmation
    confirm = input("I don't have an answer for this. Should I create a support ticket? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        return "Okay, no ticket was created."

    ticket_id = create_support_ticket(user_name=user_name, issue=user_question)
    if ticket_id:
        return f"Support ticket #{ticket_id} created successfully. Our team will respond soon."
    else:
        return "Failed to create a support ticket. Please try again later."


# Check Ticket Status Tool
@tool
def check_ticket_status_tool(ticket_id: str) -> str:
    """
    Check the status of a ticket and return human response if available.
    """
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        return f"Ticket {ticket_id} not found."
    
    status = ticket["status"]
    issue = ticket["issue"]
    response = ticket.get("response")

    if response:
        return f"Ticket {ticket_id} ({status})\nIssue: {issue}\nResponse: {response}"
    else:
        return f"Ticket {ticket_id} ({status})\nIssue: {issue}\nResponse: Pending from support."


# List of all available tools
SUPPORT_AGENT_TOOLS = [
    query_database_tool,
    query_rag_tool, 
    create_support_ticket_tool,
    check_ticket_status_tool
]
