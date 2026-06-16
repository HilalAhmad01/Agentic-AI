# %% Cell 1: Imports
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph

# %% Cell 2: Load environment variables
load_dotenv()

# %% Cell 3:
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="deepseek-ai/deepseek-v4-pro", api_key=api_key)


# %% Cell 4: State definition
class state(TypedDict):
    query: str
    llm_answer: str
    blog_post: str


# %% Cell 5: function definition


def get_llm_response(state: state) -> state:
    query = state["query"]
    prompt = f"generate a detailed outline for the topic provided by the user: {query}"
    response = llm.invoke(prompt)
    print(response.content)
    return {"llm_answer": response.content}


def generate_blog(state: state) -> state:
    llm_answer = state["llm_answer"]
    prompt = f"generate a small blog post based on the outline provided by the LLM: {llm_answer}"
    response = llm.invoke(prompt)
    return {"blog_post": response.content}


# %% Cell 6: Define the graph
graph = StateGraph(state)
graph.add_node("get_llm_response", get_llm_response)
graph.add_node("generate_blog", generate_blog)
graph.add_edge(START, "get_llm_response")
graph.add_edge("get_llm_response", "generate_blog")
graph.add_edge("generate_blog", END)
compiled = graph.compile()

# %% Cell 7: Run the graph
initial_state = {"query": "ai data centres"}
result = compiled.invoke(initial_state)
print(result)
