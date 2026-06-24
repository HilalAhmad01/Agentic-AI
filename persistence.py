# %% cell 1
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

# %% Cell 2: Load environment variables
load_dotenv()

# %% Cell 3:
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)


# %% Cell 4:
class joke_state(TypedDict):
    topic: str
    joke: str
    explanation: str


# %% Cell
def generate_joke(state: joke_state):
    topic = state["topic"]
    prompt = f"Tell me a joke about {topic} "
    response = llm.invoke(prompt).content
    return {"joke": response}


# %% Cell
def explain_joke(state: joke_state):
    joke = state["joke"]
    prompt = f"generate an explanation for the joke: {joke}\n"
    response = llm.invoke(prompt)
    return {"explanation": response.content}


# %% Cell
graph = StateGraph(joke_state)
graph.add_node("generate_joke", generate_joke)
graph.add_node("explain_joke", explain_joke)
graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "explain_joke")
graph.add_edge("explain_joke", END)
workflow = graph.compile(checkpointer=InMemorySaver())

# %% Cell
workflow

# %% Cell
config1 = {"configurable": {"thread_id": "1"}}
initial_state = {"topic": "pizza"}
workflow.invoke(initial_state, config=config1)
# %% cell idk what
workflow.get_state(config=config1)
# %% cell
list(workflow.get_state_history(config1))
# %% cell
config2 = {"configurable": {"thread_id": "2"}}
initial_state = {"topic": "pasta"}
workflow.invoke(initial_state, config=config2)
# %% cell
list(workflow.get_state_history(config2))
