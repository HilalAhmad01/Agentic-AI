import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def chat_with_bot(user_message: str, thread_id: str = "default"):
    """Send a message to the chatbot and get a response."""
    input_message = HumanMessage(content=user_message)
    response = chatbot.invoke(
        {"messages": [input_message]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return response["messages"][-1].content


sqlite3.connect(database="chatbot_checkpoints.sqlite", check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(
    sqlite3.connect("chatbot_checkpoints.sqlite", check_same_thread=False)
)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        configurable = checkpoint.config.get("configurable", {})
        if "thread_id" in configurable:
            all_threads.add(configurable["thread_id"])

    return list(all_threads)
