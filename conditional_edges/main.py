import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def node_run_agent(state: AgentState) -> AgentState:
    print("I am agent!")
    state["agent_outcome"] = "agent_outcome1"
    return state


def node_execute_tools(state: AgentState):
    print("I am tools!")
    state["agent_outcome"] = "tool1"
    return state


def should_continue(state: AgentState):
    if state["agent_outcome"] == "agent_outcome":
        return "end"
    else:
        return "continue"


def should_continue_tool(state: AgentState):
    if state["agent_outcome"] == "tool":
        return "end"
    else:
        return "continue_tool"


workflow = StateGraph(AgentState)
workflow.add_node("agent", node_run_agent)
workflow.add_node("action", node_execute_tools)
workflow.set_entry_point("agent")

# workflow.set_finish_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "action",
    should_continue_tool,
    {
        "continue_tool": END,
        "end": END,
    },
)

# workflow.add_edge("action", "agent")
app = workflow.compile()
print(app.get_graph().draw_mermaid())


inputs = {"input": "what is wether in maharashtra  pune", "chat_history": []}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("------------------------------")
