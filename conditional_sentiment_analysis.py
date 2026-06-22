# %% Cell 1: Load environment variables
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import Literal
from pydantic import BaseModel, Field

from essay_review import workflow

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
    diagnosis: str


# %% Cell 4:
class SentimentExtractionSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The overall sentiment of the review"
    )


# 2. Update your node function
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

    if state["sentiment"] == "positive":
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
graph.add_node("check_sentiment", check_sentiment)
graph.add_node("positive_response_output", positive_response_output)
graph.add_node("run_diagnosis_on_negative_response", run_diagnosis_on_negative_response)
graph.add_node("negative_reponse_output", negative_reponse_output)
graph.add_edge(START, "check_sentiment")
graph.add_edge("check_sentiment", "positive_response_output")
graph.add_edge("check_sentiment", "run_diagnosis_on_negative_response")
graph.add_edge("run_diagnosis_on_negative_response", "negative_reponse_output")
graph.set_finish_point("negative_reponse_output")
# %% Compile the graph
workflow = graph.compile()

# %% graph overview
workflow
# %% start this shit
initial_state = {
    "review": """ First of all, I would like to talk about my Realme Narzo 80 Pro 5G. Overall, it is a very good phone in the 20–22k INR price range. I have been using this phone since 17 August 2025, and today is 28 November 2025—almost 4 months. The gaming performance is very good. The camera quality is also very good in bright environments such as sunlight or outdoors, especially when taking photos of people. Nature photography is also decent. Also sound quality is also good , i'm reading some negative comments , but i want to tell to whomever reading this review , The phone has dual stereo speaker support also , with 300% voice boost .

I have the 8 GB RAM + 128 GB storage variant. I strongly suggest buying the 256 GB storage variant because in the 128 GB version, a large portion of the storage is taken by the system—around 30–40 GB. So, you only get about 60–70 GB of usable space, which is quite low for modern usage. Even though the phone is advertised as 128 GB, you don’t actually get the full 128 GB.

In terms of performance, I did not face any heating issues. The phone stays cool during normal and even heavy tasks. It only heats slightly while charging, which is completely normal for fast-charging phones. The connectivity is also very good, with strong Wi-Fi and Bluetooth performance. Storage transfer speeds are fast as well, thanks to UFS 3.1 storage.

I also appreciate the battery capacity, which is quite good for modern phones (6000 mAh). The battery lasts around 20–21 hours with moderate usage such as YouTube, Instagram Reels, or studying on the PW app. However, I have one issue with the screen. Although the display is very bright, easy to use in sunlight, and has excellent color quality, I would advise avoiding phones with curved displays. If a curved screen breaks after dropping on a cemented road or anywhere else, it is very difficult to find a duplicate replacement screen in the market.

Additionally, you cannot expand the memory using an SD card, and there is no audio jack on this phone. On the positive side, the 80W charging is amazing—it charges the phone up to 25% in just 10 minutes and more than 60% in under 30 minutes.

Overall, the phone is very good, but I still recommend checking out the Realme 15x. It offers almost the same features and is even better in some aspects because it doesn’t have a curved display. The Realme 15x also comes with 60W SUPERVOOC charging and a massive 7000 mAh battery, which is extremely impressive and lasts even longer."""
}

workflow.invoke(initial_state)
