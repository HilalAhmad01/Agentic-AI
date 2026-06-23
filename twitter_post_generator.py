# %% cell 1
import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# %% cell 2
load_dotenv()

# %% Cell 3:
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)

generator_llm = llm
evaluator_llm = llm
optimizer_llm = llm


# %% Cell 4:
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result."
    )
    feedback: str = Field(..., description="feedback for the tweet.")


# %% Cell 5:
structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)


# %% Cell 6:
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


# %% Cell 7:
def generate_tweet(state: TweetState):

    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(
            content=f"""
Write a short, original, and hilarious tweet on the topic: "{state["topic"]}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
"""
        ),
    ]

    # send generator_llm
    response = generator_llm.invoke(messages).content

    # return response
    return {"tweet": response, "tweet_history": [response]}


# %% Cell 8:
def evaluate_tweet(state: TweetState):

    # prompt
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
        ),
        HumanMessage(
            content=f"""
Evaluate the following tweet:

Tweet: "{state["tweet"]}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?
2. Humor – Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness – Is it short, sharp, and scroll-stopping?
4. Virality Potential – Would people retweet or share it?
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
"""
        ),
    ]

    response = structured_evaluator_llm.invoke(messages)

    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback,
        "feedback_history": [response.feedback],
    }


# %% Cell 9:
def optimize_tweet(state: TweetState):

    messages = [
        SystemMessage(
            content="You punch up tweets for virality and humor based on given feedback."
        ),
        HumanMessage(
            content=f"""
Improve the tweet based on this feedback:
"{state["feedback"]}"

Topic: "{state["topic"]}"
Original Tweet:
{state["tweet"]}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
"""
        ),
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state["iteration"] + 1

    return {"tweet": response, "iteration": iteration, "tweet_history": [response]}


# %% Cell 10:
def route_evaluation(state: TweetState):

    if (
        state["evaluation"] == "approved"
        or state["iteration"] >= state["max_iteration"]
    ):
        return "approved"
    else:
        return "needs_improvement"


# %% Cell 11:
graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate", route_evaluation, {"approved": END, "needs_improvement": "optimize"}
)
graph.add_edge("optimize", "evaluate")

workflow = graph.compile()

workflow
# %% Cell 12:
initial_state = {"topic": "Indian Railways", "iteration": 1, "max_iteration": 5}
result = workflow.invoke(initial_state)
# %% Cell 13:
result
for tweet in result["tweet_history"]:
    print(tweet)
