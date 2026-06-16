# %% Cell 1: Imports
import os
from sys import int_info
from typing import TypedDict

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from typing_extensions import final

# %% Cell 2: Load environment variables
load_dotenv()

# %% Cell 3:
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="deepseek-ai/deepseek-v4-pro", api_key=api_key)


# %% Cell 4:
class state(TypedDict):
    query: str
    answer: str


def user_query(state: state) -> state:
    question = state["query"]
    prompt = f"answer the following question: {question} and give the answer in a clean and concise manner"
    answer = llm.invoke(prompt).content
    state["answer"] = answer
    return state


# %% Cell 5:
graph = StateGraph(state)
graph.add_node("user_query", user_query)
graph.add_edge(START, "user_query")
graph.add_edge("user_query", END)
app = graph.compile()

# %% Cell 6:
initial_state = {"query": "what is the use of init function in python?"}
final_state = app.invoke(initial_state)
print(final_state)
