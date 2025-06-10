from typing import TypedDict, Sequence, Annotated,List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import os
import json

load_dotenv()

# ---- State Definition ----
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---- Tool Definition ----
# @tool
# def add(a: int, b: int):
#     """
#     This adds two numbers a and b and returns the answer.
#     """
#     return a + b

# tools = [add]
@tool
def sum_numbers(numbers: List[int]) -> int:
    """Calculates the sum of a list of numbers."""
    return sum(numbers)
tools = [sum_numbers]

# ---- Model Binding ----
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", temperature=0.7
).bind_tools(tools)


# ---- Agent Node ----
def model_call(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(content="You are my AI model. Answer the query to the best of your knowledge.")
    response = model.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}


# ---- Conditional Logic ----
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

# ---- Graph Construction ----
graph = StateGraph(AgentState)
graph.add_node("our_Agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_Agent")

graph.add_conditional_edges(
    "our_Agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_Agent")

app = graph.compile()

# ---- Save Graph Image ----
with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

print("âœ… Graph image saved as 'graph.png'")


# ---- Function: Print Full Raw Message Details ----
def print_all_raw_messages(state: AgentState):
    print("\n\n=== ðŸ” RAW MESSAGE STRUCTURE ===")
    for idx, msg in enumerate(state["messages"]):
        print(f"\n--- Message {idx + 1} ---")
        try:
            if hasattr(msg, 'dict'):
                print(json.dumps(msg.dict(), indent=2))
            elif hasattr(msg, '__dict__'):
                print(json.dumps(msg.__dict__, indent=2, default=str))
            else:
                print(repr(msg))
        except Exception as e:
            print(f"Error printing message: {e}")
            print(repr(msg))


# ---- Run Agent ----
inputs = {
    "messages": [HumanMessage(content="add 1,2,3,1,2,3 and also this one to it 4,5,6,5,4,6")]
}

final_state = None
print("\n=== ðŸ’¬ Streaming Output ===")
for state in app.stream(inputs, stream_mode="values"):
    final_state = state
    last_message = state["messages"][-1]
    try:
        last_message.pretty_print()
    except AttributeError:
        print(last_message)

# ---- Print Raw Messages After Execution ----
print_all_raw_messages(final_state)