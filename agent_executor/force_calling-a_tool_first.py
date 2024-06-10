import operator
from typing import Annotated, TypedDict, Union
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent

# from langchain_community.llms import Ollama


tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/openai-functions-agent")
# llm = Ollama(model="llama3")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
agent_runnable = create_openai_functions_agent(llm, tools, prompt)


class AgentState(TypedDict):
    """State definition including required fields."""

    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


def agent_node(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    state["agent_outcome"] = agent_outcome
    return state


def action_node(state: AgentState):
    state["tool_call"] = "tool_call"
    state["messages"] = state["messages"] + " | Tool processed"
    return state


def should_continue(state: AgentState):
    """Determines if the workflow should continue or end."""
    if state["agent_message"] == "agent_message":
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node(
    "agent",
    agent_node,
)
# workflow.add_node(
#     "action",
#     action_node,
# )
workflow.set_entry_point("agent")

workflow.set_finish_point("agent")
# workflow.add_edge("action", END)

# workflow.add_conditional_edges(
#     "agent",
#     should_continue,
#     {
#         "end": END,
#         "continue": "action",
#     },
# )
app = workflow.compile()
# Generate the PNG binary data
mermaid_png = app.get_graph().draw_mermaid_png()

# Save the PNG binary data to a file
with open("workflow_diagram.png", "wb") as file:
    file.write(mermaid_png)

print("The workflow diagram has been saved as 'workflow_diagram.png'.")

inputs = {"input": "what is the weather in maharashtra?", "chat_history": []}
# res = app.invoke(inputs)
for s in app.stream(input=inputs):
    print("response:", s)
