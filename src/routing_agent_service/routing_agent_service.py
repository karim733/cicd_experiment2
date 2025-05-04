import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import Annotated

from fastapi import FastAPI
import uvicorn
import argparse

from dotenv import load_dotenv
#_ = load_dotenv('../secrets/.env')


global_config = {}

ROUTING_PROMPT="""
Route the input to: 
a) crisis - if there is TRUE mental health emergency (eg. suicideal ideation, self harm, etc.) either explicitly stated
or implied in the input
b) rag - if the input is a specific QUESTION about techniques for defeating guilt, dealing with anger, overcoming perfectionism,
dealing with criticism, and dealing with addiction.  The input must include specific question about ways for handling
one of these topics - questions that would benefit with factual answers from a self-help book.  The input MUST
BE A QUESTION, not a statement.
c) fine_tuned (for all other cases, and for general conversation flow)
"""


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["fine_tuned", "crisis", "rag"] = Field(
        None, description="The next step in the routing process"
    )

class State(TypedDict):
    input: str
    decision: str
    output: str
    messages: Annotated[list, add_messages]
    

llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

router = llm.with_structured_output(Route)

def convert_to_fine_tuned_format(messages):
    content = []
    for m in messages:
      if isinstance(m,HumanMessage):
        content.append({"from":"patient","value":m.content})
      elif isinstance(m,AIMessage):
        content.append({"from":"therapist","value":m.content})
    return content


def fine_tuned(state: State):
    import requests
    #ft_endpoint=global_config["ft_endpoint"]
    #fine_tuned_url=]"http://127.0.0.1:8001/fine_tuned_chat"
    ft_endpoint="http://ft_test_service:8002/sft/inference"

    data = {
            "messages": convert_to_fine_tuned_format(state["messages"])
            }
    #print(data)
    print('*********************FINE_TUNED******************************')
    response = requests.post(ft_endpoint, headers={"Content-Type": "application/json"}, json=data)
    print(response.json())
    return {"messages":AIMessage(content=response.json()[-1]["value"]),"output": "fine_tuned"}
    

def crisis(state: State):
    print('*********************Crisis-State**********************')
    #result = llm.invoke(state["input"])
    #return {"output": result.content}
    CRISIS_MESSAGE="CRISIS: You need professional help!  This bot can't help you!"
    return {"messages": AIMessage(content=CRISIS_MESSAGE),"output": "crisis"}

def rag(state: State):
    import requests
    print('*********************RAG******************************')
    data = {"question": state["messages"][-1].content}
    response = requests.post("http://rag_test_service:8003/rag/str", params=data)
    #response = requests.post(global_config["rag_endpoint"], params=data)
    #print(response.text, '\n')
    return {"messages":AIMessage(content=response.text),"output": "rag"}


def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(content=ROUTING_PROMPT),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "fine_tuned":
        return "fine_tuned"
    elif state["decision"] == "rag":
        return "rag"
    elif state["decision"] == "crisis":
        return "crisis"


router_builder = StateGraph(State)
router_builder.add_node("fine_tuned", fine_tuned)
router_builder.add_node("crisis", crisis)
router_builder.add_node("rag", rag)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
                    "llm_call_router",
                    route_decision,
                    {  # Name returned by route_decision : Name of next node to visit
                        "fine_tuned": "fine_tuned",
                        "rag": "rag",
                        "crisis": "crisis",
                    },
)
router_builder.add_edge("fine_tuned", END)
router_builder.add_edge("crisis", END)
router_builder.add_edge("rag", END)

memory = MemorySaver()
router_workflow = router_builder.compile(checkpointer=memory)
app = FastAPI()


from fastapi import FastAPI
from pydantic import BaseModel
class ChatRequest(BaseModel):
    id: str
    user_input: str

    
@app.post('/chat')
def chat(req : ChatRequest):
    config = {"configurable": {"thread_id": req.id}}
    input_messages = [HumanMessage(req.user_input)]
    
    output = router_workflow.invoke({"messages": input_messages,"input":req.user_input}, config)
    #print(f"****chat endpoint*****: {output["messages"][-1]}")
    return output["messages"][-1].content
