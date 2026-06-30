# %% cell 1 imports
import os
from typing import Annotated, TypedDict

import cohere
import langchain_cohere
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# %% cell
load_dotenv()
api_key = os.environ.get("NVDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=api_key)

# %% cell
loader = PyPDFLoader("machine learning.pdf")
docs = loader.load()

# %% cell
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# %% cell
cohere_embeddings = CohereEmbeddings(
    model="embed-v4.0", cohere_api_key=os.getenv("COHERE_API_KEY")
)

vectorstore = FAISS.from_documents(chunks, cohere_embeddings)

# %% cell
len(chunks)
vectorstore
# %% cell
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# %% cell
@tool
def rag_tool(query):
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {"query": query, "context": context, "metadata": metadata}


# %% cell
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


# %% cell
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# %% cell


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# %% cell
tool_node = ToolNode(tools)

# %% cell
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

# %% cell
chatbot
# %% cell
result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=("Using the pdf notes, explain what is machine learning ")
            )
        ]
    }
)
print(result["messages"][-1].content)
