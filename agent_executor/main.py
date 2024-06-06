import operator
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt.tool_executor import ToolExecutor

load_dotenv()


class AgentState(TypedDict):
    """Agent State"""

    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# create agent
tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
agent_runnable = create_openai_functions_agent(llm, tools, prompt)


# Create node for agent
def run_agent(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolExecutor(tools)


def execute_tools(state: AgentState):
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}


def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"


# create graph
workflow = StateGraph(AgentState)
# Define two nodes we will cycle between

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")

workflow.set_finish_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
app = workflow.compile()
print(app.get_graph().draw_mermaid())
# display(Image(app.get_graph(xray=True).draw_mermaid_png()))
inputs = {"input": "what is wether in maharashtra  pune", "chat_history": []}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("------------------------------")
