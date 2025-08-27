from typing import List, TypedDict
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from agent_tools import SUPPORT_AGENT_TOOLS
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")


# Define Graph State
class AgentState(TypedDict):
    chat_history: List[BaseMessage]
    agent_outcome: List[tuple]  


# Initialize LLM
llm = init_chat_model("groq:llama3-70b-8192", temperature=0)

# Agent system prompt
SYSTEM_PROMPT = """You are an autonomous customer support agent. Analyze each query and choose the appropriate tools.

AVAILABLE TOOLS:
1. query_database_tool - For team members, contact info, FAQs, and structured data
2. query_rag_tool - For documentation, policies, procedures, general information  
3. create_support_ticket_tool - When you cannot answer or need human expertise
4. check_ticket_status_tool - For checking existing ticket status

DECISION FRAMEWORK:
- People questions → query_database_tool (e.g., "who is alice")
- Policy/documentation → query_rag_tool (e.g., "refund policy", "pricing")
- Complex/unknown topics → create_support_ticket_tool (e.g., "custom integration")
- Ticket status → check_ticket_status_tool (e.g., "status of TKT-ABC123")

Think step-by-step:
1. Analyze the user's query
2. Choose the most appropriate tool(s)
3. Execute the tool(s)
4. Synthesize the response

Always be helpful, professional, and concise. If you need to use multiple tools, do so."""

# Define Nodes


def run_agent(state: AgentState):
    print("---RUN AGENT---")
    chat_history = state["chat_history"]
    agent_scratchpad = state["agent_outcome"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, SUPPORT_AGENT_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=SUPPORT_AGENT_TOOLS, verbose=True)

    result = agent_executor.invoke(
        {
            "input": chat_history[-1].content,
            "chat_history": chat_history[:-1],
            "agent_scratchpad": agent_scratchpad,
        }
    )

    if isinstance(result, BaseMessage):
        # Agent returned a direct response without tool calls
        return {"agent_outcome": [], "chat_history": state["chat_history"] + [result]}
    elif isinstance(result, dict):
        # Agent invoked tools or returned a dictionary output
        output_content = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        return {
            "agent_outcome": intermediate_steps,
            "chat_history": state["chat_history"] + [AIMessage(content=output_content)],
        }
    else:
        # Fallback for unexpected result types
        return {
            "agent_outcome": [],
            "chat_history": state["chat_history"] + [AIMessage(content=str(result))],
        }


# Create a ToolNode to handle tool execution
tool_node = ToolNode(SUPPORT_AGENT_TOOLS)


# Define Edges (Transitions)
def should_continue(state: AgentState) -> str:
    print("---DECIDE TO CONTINUE---")
    last_message = state["chat_history"][-1]

    if (
        "tool_calls" in last_message.additional_kwargs
        and last_message.additional_kwargs["tool_calls"]
    ):
        return "continue_tool_call"
    else:
        return "end"


# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("tools", tool_node)  # Use the ToolNode here

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue_tool_call": "tools", "end": END}
)

workflow.add_edge("tools", "agent")

app = workflow.compile()


def process_graph_with_agent(
    user_input: str, conversation_history: List[BaseMessage] = None
):
    initial_state = {
        "chat_history": conversation_history if conversation_history else [],
        "agent_outcome": [],  # Initialize agent_outcome as an empty list
    }
    initial_state["chat_history"].append(HumanMessage(content=user_input))

    for s in app.stream(initial_state):
        if "__end__" not in s:
            pass  # We will handle printing in main.py

    final_state = app.invoke(initial_state)
    return final_state["chat_history"][-1]
