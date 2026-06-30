# %% cell 1
import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph

# Load environment variables
load_dotenv()

# %% cell 2
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)


# %% cell 3
# Node
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# %% cell 4
# Graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# %% cell 5
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

# %% cell 6
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 1 (remembers)
    t1 = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke(
        {"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, t1
    )
    out1 = graph.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]}, t1
    )
    print("Thread-1:", out1["messages"][-1].content)


# %% cell 7
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 2 (fresh)
    t2 = {"configurable": {"thread_id": "thread-2"}}
    out2 = graph.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]}, t2
    )
    print("Thread-2:", out2["messages"][-1].content)

# %% cell 8


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"
t1 = {"configurable": {"thread_id": "thread-1"}}

with PostgresSaver.from_conn_string(DB_URI) as cp:
    g = builder.compile(checkpointer=cp)

    snap = g.get_state(t1)  # <-- pulls from Postgres
    msgs = snap.values.get("messages", [])
    print("Last message:", msgs[-1].content if msgs else None)
