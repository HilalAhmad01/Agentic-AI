# %% Cell 1: Imports
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core import language_models
from langchain_core.tools import base
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# %% Cell 2: Load environment variables
load_dotenv()

# %% Cell 2: Load environment variables
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)


# %% Cell 4:
class evaluation_schema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay.")
    score: int = Field(
        description="score out of 10 based on the essay content.", gt=0, lt=11
    )


# %% Cell 5:
structured_model = llm.with_structured_output(evaluation_schema)

# %% Cell 6:
essay = """The title might sound outrageous, but this is something that is proven daily to us, but we choose to ignore it and forget about it, and no, this is not coming from a racist westerner on Twitter.

I mainly want to highlight how India is treated as an experiment lab by the whole world and how our government enables such behavior without a second thought and sometimes packages such behavior into “victories” (like with India’s approach to AI, which we will talk about later).

Before we talk about how AI is reinforcing this destructive behavior, we need to cover the basics so we all are educated about how this is not some one-off blunder but a reoccurring thing. The most common example of this behavior is with food.

Thanks to social media, most people are aware by now about companies often selling the same products with different ingredients in India compared to other countries; for example, in India, many popular chocolates use vegetable fat or palm oil instead of cocoa butter. This is legally allowed here, but in the UK, Australia, and much of Europe, chocolate must meet stricter cocoa butter requirements.

This is just the tip of the iceberg , lets take a look at two such more examples -


Banned in Europe, ‘Compliant’ in India
A not-so-well-known example of this happened recently when an Italian chemical plant that produced PFAS was forced to shut down after contaminating soil and water in an area inhabited by 350,000 people because residents were reporting alarmingly high levels of PFAS, also known as ‘forever chemicals,’ in their blood. These chemicals have been linked to infertility, cardiac diseases, and cancer.

However, where other countries ban such factories to protect their citizens, Indian companies view this as a money-making opportunity, as the plant was then purchased by an Indian company and reopened south of Mumbai. Thankfully, there was some backlash over this, which prompted a response from the company. The response was a complete refusal of the claims made by multiple investigations and news articles online. The company states that the facility operates in complete compliance with all applicable Indian environmental, safety, and regulatory requirements.

An MPCB (Maharashtra Pollution Control Board) official said, “The company has got its paperwork in place. Forever chemicals are not banned in India. When it comes to causing pollution, if we find any anomaly, then we will act on it, give a show-cause notice, a closure notice, and so on.”

“Not banned” ≠ safe. PFAS are called “forever chemicals” for a reason they persist in the environment for thousands of years, leach into water/soil, and are hard (and insanely expensive) to monitor or clean up.Paperwork doesn’t equal real protection. Miteni (the Italian chemical plant) had permits too until the scandal exploded. The same electrochemical fluorination tech that caused Italy’s crisis is now here.The Indian chemical plant insists it is fully compliant and regulated in other countries, but the optics are terrible. Why is India importing Europe’s toxic legacy?


Indias’s approach to AI
While other countries are busy trying to build the best AI model in the world, India has a different approach to AI, or let’s just say a lack thereof. See, AI development has almost become necessary in this day and age, and countries are willing to do anything to win this race. Both the US and China treat AI as a national strategic priority and actively support their leading AI companies through coordinated government policies, funding, infrastructure, talent programs, and strategic controls. While this growth is normal and necessary, AI’s rapid development faces a major roadblock ahead: energy-intensive data centers.

These data centers are the core of training, deploying, and running these AI models, and these consume a massive amount of electricity and water to operate. Even a mid-sized data center consumes as much water as a small town, while larger ones require up to 5 million gallons of water every day, as much as a city of 50,000 people. In terms of electricity, they consume enough electricity that companies are building nuclear plants to power future data centers.

This has been a growing problem in the US, where data centers more than doubled between 2018 and 2021. All this electricity doesn’t come free, and the cost is passed down to the customer, with some people reporting a 2x increase in their electricity bills. This expansion of data centers at the expense of the general public is not something that would last long, as the general sentiment towards AI and especially data centers is growing overly negative now.

A recent incident supporting this overly growing negative sentiment comes from New Brunswick, New Jersey, where local authorities blocked the construction of a data center on a plot of land slated for redevelopment, instead requiring that a park be built on the site.

Now what do companies do when a venture is blocked or faces strong local resistance over health or environmental concerns? That’s right, they shift their operations to India, where they face the least amount of regulatory friction and benefits like tax breaks and cheap resources.

This is India’s approach towards AI: instead of building our own sovereign models, we are spending billions of dollars to construct power-hungry data centers for foreign companies on our own land.At a time when Delhi, Bengaluru, and Chennai are already facing acute water shortages, these data centers will only amplify the crisis.

India is building Americas’s tech
Recently India hosted one of the biggest AI summits, where CEOs of every major AI firm from the US attended the event. The whole event was touted as a major success (even though it was just a glorified marketing event for OpenAI, Claude, Google, etc.). Alongside this major billion-dollar deals were signed with OpenAI and Google. On the surface, this looks like a major win for India, as foreign investment is always welcome. But the optimism proves short-lived once you discover that the bulk of these deals are focused almost exclusively on building power hungry AI data centers.
Yes, the same data centers that are facing major pushback in the US are now coming to India.

Let’s take a look at two of the biggest deals that were signed in “partnership” with US companies -


Adani Group’s $100 billion AI data center commitment in partnership with Google

TCS SIGNS OPENAI AS DATA CENTER CUSTOMER


These partnerships may look like a win for India because of the massive capital inflow, but in practice India is primarily taking on the heavy lifting: building and powering the energy-intensive data centers while American companies retain ownership of the frontier AI models and capture most of the long-term strategic value. We bear the infrastructure risks and environmental costs yet still depend on foreign models for the highest-value applications.

India has emerged as the world’s largest consumer of AI, and foreign companies want to keep it that way. If India develops its own competitive models, they risks losing a massive portion of their user base. In the end, when regulations become a problem back home, countries seek India like a safe haven. The government/big companies profit from such deals by putting the livelihood of the common man at risk, whether it’s the poison they feed us or the poison or the poison producing factories they import.

The lives of Indians are always compromised."""
result = structured_model.invoke(essay)
print(result)


# %% Cell 7:
class essay_state(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


# %% Cell 8:
def evaluate_language(state: essay_state):
    essay = state["essay"]
    prompt = f"evluate the language quality of the essay and provide a feedback and assign it a score out of 10 \n {essay}"
    result = structured_model.invoke(prompt)
    return {"language_feedback": result.feedback, "individual_scores": [result.score]}


# %% Cell 9:
def evaluate_analysis(state: essay_state):
    essay = state["essay"]
    prompt = f"evluate the depth of analysis of the essay and provide a feedback and assign it a score out of 10 \n {essay}"
    result = structured_model.invoke(prompt)
    return {"analysis_feedback": result.feedback, "individual_scores": [result.score]}


# %% Cell 10:
def evaluate_clarity(state: essay_state):
    essay = state["essay"]
    prompt = f"evluate the clarity of the essay and provide a feedback and assign it a score out of 10 \n {essay}"
    result = structured_model.invoke(prompt)
    return {"clarity_feedback": result.feedback, "individual_scores": [result.score]}


# %% Cell 11:
def evaluate_overall(state: essay_state):
    # summary feedback
    prompt = f"Based on the following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback - {state['clarity_feedback']}"
    overall_feedback = llm.invoke(prompt).content

    # avg calculate
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


# %% Cell 12:
graph = StateGraph(essay_state)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_clarity)
graph.add_node("evaluate_overall", evaluate_overall)

# edges
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "evaluate_overall")
graph.add_edge("evaluate_analysis", "evaluate_overall")
graph.add_edge("evaluate_thought", "evaluate_overall")

graph.add_edge("evaluate_overall", END)

workflow = graph.compile()

# %% Cell 13:
workflow

# %% Cell 14:
initial_state = {"essay": essay}

workflow.invoke(initial_state)
