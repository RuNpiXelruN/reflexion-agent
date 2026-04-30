# from dotenv import load_dotenv
# load_dotenv()

from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2


def draft_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    if count_tool_visits >= MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop)

graph = builder.compile()

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the best way to diy a sandstone wall that's about 600mm high using many sized stones, the largest being around 400mm wide and 300mm high?",
                }
            ]
        }
    )

    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])

    print(res)