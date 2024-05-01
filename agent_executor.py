import os
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated, Union
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import random


load_dotenv(".env")

openAI = os.getenv("OPENAI_API_KEY")
polygon = os.getenv("POLYGON_API_KEY")

langsmith = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent_executor"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))

# prompt from langChain hub
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# StateAgent
class AgentState(TypedDict):
   input: str
   # list of previous messages in the conversation
   chat_history: list[BaseMessage]
   # outcome of a given call to the agent
   agent_outcome: Union[AgentAction, AgentFinish, None]
   # list of actions and corresponding observations
   # 'add' is so that the state is added to the existing values, (it does not overwrite it)
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# DEFINE THE TOOLS
@tool("lower_case", return_direct=True)
def to_lower_case(input:str) -> str:
    """Returns the input as all lower case."""
    return input.lower()

@tool("random_number", return_direct=True)
def random_number_maker(input:str) -> str:
    """Returs a random number between 0-100."""
    return random.randint(0, 100)

tools = [to_lower_case, random_number_maker]


# DEFINE THE AGENT
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

agent = RunnablePassthrough.assign(
    agent_outcome = agent_runnable
)

inputs = {"input": "give me a random number and then write it in words and make it lower case.",
          "chat_history": [],
          "intermediate_steps":[]}

agent_outcome = agent_runnable.invoke(inputs)

# GIVES ABILITY TO EXECUTE THE TOOLS
tool_executor = ToolExecutor(tools)

# DEFINE NODES
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}


# DEFINE THE FUNCTION TO EXECUTE THE TOOLS
def execute_tools(data):
    # Get the most recent agent_outcome
    agent_action = data['agent_outcome']
    # Execute the tool
    output = tool_executor.invoke(agent_action)
    # print(f"The agent action is {agent_action}")
    # print(f"The tool result is: {output}")
    # Return the output
    return {"intermediate_steps": [(agent_action, str(output))]}


# DEFINE LOGIC THAT WILL BE USED TO DETERMINE WHICH CONDITIONAL EDGE TO USE 
def should_continue(data):
    # If the agent outcome is AgentFinish, then exit
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    # Otherwise an AgentAction is returned
    else:
        return "continue"

# DEFINE THE GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge('action', 'agent')


#
# COMPILE GRAPH
app = workflow.compile()

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")

output = app.invoke(inputs)
print(output.get("agent_outcome").return_values['output'])