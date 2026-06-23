# %% Cell 1: Load environment variables
import os
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import Literal
from pydantic import BaseModel, Field

# %% Cell 2: Load environment variables
load_dotenv()

# %% Cell 3:
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)


class sentiment_state(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    negative_reponse: str
    positive_response: str
    diagnosis: Any


# %% Cell 4:
class SentimentExtractionSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The overall sentiment of the review"
    )


def find_sentiment(state: sentiment_state) -> sentiment_state:
    review = state["review"]

    # Pass the CORRECT schema to the LLM
    structured_llm = llm.with_structured_output(SentimentExtractionSchema)

    # The LLM will now return an object with a .sentiment attribute
    result = structured_llm.invoke(f"Analyze the sentiment of this review: {review}")

    # This will now work perfectly!
    return {"sentiment": result.sentiment}


# %% Cell :
def check_sentiment(
    state: sentiment_state,
) -> Literal["positive_response_output", "run_diagnosis_on_negative_response"]:
    sentiment = state.get("sentiment")
    if sentiment == "positive":
        return "positive_response_output"
    else:
        return "run_diagnosis_on_negative_response"


# %% Cell 5:
def positive_response_output(state: sentiment_state) -> sentiment_state:
    review = state["review"]
    positive_response = llm.invoke(
        f"Analyze the following review and provide a positive response: {review}"
    ).content
    return {"positive_response": positive_response}


# %% Cell 6:
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )


def run_diagnosis_on_negative_response(state: sentiment_state) -> sentiment_state:
    review = state["review"]
    structured_model = llm.with_structured_output(DiagnosisSchema)
    prompt = f"Analyze the following user review and extract the required diagnosis fields:\n\n{review}"
    diagnosis = structured_model.invoke(prompt)
    return {"diagnosis": diagnosis}


# %% Cell 7:
def negative_reponse_output(state: sentiment_state) -> sentiment_state:
    diagnosis = state["diagnosis"]
    negative_response = llm.invoke(
        f"Given the following diagnosis, provide a appropirate response: {diagnosis}"
    ).content
    return {"negative_reponse": negative_response}


# %% Cell
graph = StateGraph(sentiment_state)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response_output", positive_response_output)
graph.add_node("run_diagnosis_on_negative_response", run_diagnosis_on_negative_response)
graph.add_node("negative_reponse_output", negative_reponse_output)
graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("run_diagnosis_on_negative_response", "negative_reponse_output")
graph.add_edge("positive_response_output", END)
graph.add_edge("negative_reponse_output", END)

# %% Compile the graph
workflow = graph.compile()

# %% graph overview
workflow
# %% start this shit
initial_state = {
    "review": """I recently purchased the Realme Narzo Pro which comes with an advertised 80W fast charger, but unfortunately my experience with the charging performance has not been good. The charging speed is not as fast as expected, and it feels much slower compared to what is promoted by the company.

    Even when using the original 80W charger that came with the phone, the device takes a longer time to fully charge. In some cases, the charging speed drops significantly, especially after the battery reaches around 40–50%. This makes the “fast charging” feature feel inconsistent and disappointing.

    Another issue I noticed is that sometimes the phone does not maintain stable fast charging, and the charging speed keeps fluctuating. For a phone that advertises high-speed charging, I expected a much better and more reliable performance.

    Because fast charging is one of the main reasons people choose this phone, the current charging experience does not meet expectations. I hope this issue can be improved through a software update or checked by the service center.

    Overall, the phone itself is good, but the charging performance with the 80W charger has been disappointing so far and does not match the advertised fast charging capability."""
}

workflow.invoke(initial_state)
