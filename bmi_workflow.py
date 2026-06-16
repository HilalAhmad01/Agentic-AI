# %% Cell 1: Imports
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# %% Cell 2: State Schema
class BMIstate(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str


# %% Cell 3:
def calculate_bmi(state: BMIstate) -> BMIstate:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height**2)

    state["bmi"] = round(bmi, 2)
    return state


# %% Cell idk:
def catagorise_bmi(state: BMIstate) -> BMIstate:
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif bmi < 25:
        state["category"] = "Normal weight"
    elif bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"
    return state


# %% Cell 4:
graph = StateGraph(BMIstate)

graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("catagorise_bmi", catagorise_bmi)
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "catagorise_bmi")
graph.add_edge("catagorise_bmi", END)

compiled = graph.compile()


# %% Cell 5:
initial_state = {"weight": 70, "height": 1.75}
result = compiled.invoke(initial_state)
print(result)
# %% Cell 6:
from IPython.display import Image

Image(compiled.get_graph().draw_mermaid_png())
