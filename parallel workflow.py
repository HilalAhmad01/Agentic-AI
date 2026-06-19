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


class cricket_state(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strikerate: float
    bowlsperboundary: float
    boundary_percentage: float
    summary: str


# %% Cell 4: calculate_strike_rate
def calculate_strike_rate(state: cricket_state):
    sr = state["runs"] / state["balls"] * 100
    state["strikerate"] = sr
    return {"strikerate": sr}


# %% Cell 5: calculate_bowls_perboundary
def calculate_bowls_perboundary(state: cricket_state):
    bpb = state["balls"] / (state["fours"] + state["sixes"])
    state["bowlsperboundary"] = bpb
    return {"bowlsperboundary": bpb}


# %% Cell 5: calculate_perboundary_percentage
def calculate_boundary_percentage(state: cricket_state):
    bp = (state["fours"] + state["sixes"]) / state["balls"] * 100
    state["boundary_percentage"] = bp
    return {"boundary_percentage": bp}


# %% Cell 6: summary
def summary(state: cricket_state):
    summary = f"Summary: {state['runs']} runs in {state['balls']} balls, with a strike rate of {state['strikerate']:.2f} and {state['boundary_percentage']:.2f}% boundary percentage."
    state["summary"] = summary
    return {"summary": summary}


# %% cell 7: state graph

state_graph = StateGraph(cricket_state)
state_graph.add_node("calculate_strike_rate", calculate_strike_rate)
state_graph.add_node("calculate_bowls_perboundary", calculate_bowls_perboundary)
state_graph.add_node("calculate_boundary_percentage", calculate_boundary_percentage)
state_graph.add_node("summary", summary)


# %% cell 9: compile graph
state_graph.add_edge(START, "calculate_strike_rate")
state_graph.add_edge(START, "calculate_bowls_perboundary")
state_graph.add_edge(START, "calculate_boundary_percentage")
state_graph.add_edge("calculate_strike_rate", "summary")
state_graph.add_edge("calculate_bowls_perboundary", "summary")
state_graph.add_edge("calculate_boundary_percentage", "summary")
state_graph.add_edge("summary", END)
workflow = state_graph.compile()

# %% cell 10:
workflow

# %% cell 11:
initial_state = {"runs": 100, "balls": 50, "fours": 6, "sixes": 4}
workflow.invoke(initial_state)
