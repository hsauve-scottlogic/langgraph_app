import os
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langgraph.prebuilt import create_agent_executor
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import create_openai_functions_agent
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated, Union, Annotated, Any, Dict, Optional, Sequence, List, Tuple
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import random
from langchain_core.messages import HumanMessage
import asyncio

load_dotenv(".env")

openAI = os.getenv("OPENAI_API_KEY")
polygon = os.getenv("POLYGON_API_KEY")

langsmith = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph 03 Multi-agent Supervisor"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))

# prompt from langChain hub
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0, max_tokens=1024, model="gpt-3.5-turbo-0125")


# GRAPH NODE - PROGRAMMER

class Code(BaseModel):
    """Plan to follow in future"""

    code: str = Field(
        description="Detailed optmized error-free Python code on the provided requirements"
    )

code_gen_prompt = ChatPromptTemplate.from_template(
    '''**Role**: You are a expert software python programmer. You need to develop python code
**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code

*REQUIREMENT*
{requirement}'''
)
coder = create_structured_output_runnable(
    Code, llm, code_gen_prompt
)

code_ = coder.invoke({'requirement':'Generate Fibonacci series'})

# GRAPH NODE - TESTER


class Test(BaseModel):
    """Plan to follow in future"""

    Input: List[List] = Field(
        description="Input for Test cases to evaluate the provided code"
    )
    Output: List[List] = Field(
        description="Expected Output for Test cases to evaluate the provided code"
    )

test_gen_prompt = ChatPromptTemplate.from_template(
    '''**Role**: As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code. 
These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
**1. Basic Test Cases**:
- **Objective**: Basic and Small scale test cases to validate basic functioning 
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.
**Instructions**:
- Implement a comprehensive set of test cases based on requirements.
- Pay special attention to edge cases as they often reveal hidden bugs.
- Only Generate Basics and Edge cases which are small
- Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases
*REQURIEMENT*
{requirement}
**Code**
{code}
'''
)
tester_agent = create_structured_output_runnable(
    Test, llm, test_gen_prompt
)

print(code_.code)

test_ = tester_agent.invoke({'requirement':'Generate Fibonacci series','code':code_.code})
print(test_)

# GRAPH NODE - PYTHON EXECUTOR

class ExecutableCode(BaseModel):
    """Plan to follow in future"""

    code: str = Field(
        description="Detailed optmized error-free Python code with test cases assertion"
    )

python_execution_gen = ChatPromptTemplate.from_template(
    """You have to add testing layer in the *Python Code* that can help to execute the code. You need to pass only Provided Input as argument and validate if the Given Expected Output is matched.
*Instruction*:
- Make sure to return the error if the assertion fails
- Generate the code that can be execute
Python Code to excecute:
*Python Code*:{code}
Input and Output For Code:
*Input*:{input}
*Expected Output*:{output}"""
)
execution = create_structured_output_runnable(
    ExecutableCode, llm, python_execution_gen
)

code_execute = execution.invoke({"code":code_.code,"input":test_.Input,'output':test_.Output})

print(code_execute.code)

# GRAPH NODE - DEBUGGER

class RefineCode(BaseModel):

    code: str = Field(
        description="Optimized and Refined Python code to resolve the error"
    )
    

python_refine_gen = ChatPromptTemplate.from_template(
    """You are expert in Python Debugging. You have to analysis Given Code and Error and generate code that handles the error
    *Instructions*:
    - Make sure to generate error free code
    - Generated code is able to handle the error
    
    *Code*: {code}
    *Error*: {error}
    """
)
refine_code = create_structured_output_runnable(
    RefineCode, llm, python_refine_gen
)

dummy_json = {
    "code": code_execute.code,
    "error": "This is an error"
}

refine_code_ = refine_code.invoke(dummy_json)


print(refine_code_.code)

exec(refine_code_.code)

# GRAPH DESIGN

class AgentCoder(TypedDict):
    requirement: str
    code: str
    tests: Dict[str, any]
    errors: Optional[str]

def programmer(state):
    print(f'Entering in Programmer')
    requirement = state['requirement']
    code_ = coder.invoke({'requirement':requirement})
    return {'code':code_.code}

def debugger(state):
    print(f'Entering in Debugger')
    errors = state['errors']
    code = state['code']
    refine_code_ = refine_code.invoke({'code':code,'error':errors})
    return {'code':refine_code_.code,'errors':None}

def executer(state):
    print(f'Entering in Executer')
    tests = state['tests']
    input_ = tests['input']
    output_ = tests['output']
    code = state['code']
    executable_code = execution.invoke({"code":code,"input":input_,'output':output_})
    #print(f"Executable Code - {executable_code.code}")
    error = None
    try:
        exec(executable_code.code)
        print("Code Execution Successful")
    except Exception as e:
        print('Found Error While Running')
        error = f"Execution Error : {e}"
    return {'code':executable_code.code,'errors':error}

def tester(state):
    print(f'Entering in Tester')
    requirement = state['requirement']
    code = state['code']
    tests = tester_agent.invoke({'requirement':requirement,'code':code})
    #tester.invoke({'requirement':'Generate Fibonacci series','code':code_.code})
    return {'tests':{'input':tests.Input,'output':tests.Output}}

def decide_to_end(state):
    print(f'Entering in Decide to End')
    if state['errors']:
        return 'debugger'
    else:
        return 'end'

workflow = StateGraph(AgentCoder)

# Define the nodes
workflow.add_node("programmer", programmer)  
workflow.add_node("debugger", debugger) 
workflow.add_node("executer", executer) 
workflow.add_node("tester", tester) 
#workflow.add_node('decide_to_end',decide_to_end)

# Build graph
workflow.set_entry_point("programmer")
workflow.add_edge("programmer", "tester")
workflow.add_edge("debugger", "executer")
workflow.add_edge("tester", "executer")
#workflow.add_edge("executer", "decide_to_end")

workflow.add_conditional_edges(
    "executer",
    decide_to_end,
    {
        "end": END,
        "debugger": "debugger",
    },
)

# Compile
app = workflow.compile()

requirement = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.You can return the answer in any order."""

config = {"recursion_limit": 50}
inputs = {"requirement": requirement}
running_dict = {}

async def main():
    config = {"recursion_limit": 50}
    inputs = {"requirement": requirement}
    running_dict = {}

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            running_dict[k] = v
            if k != "__end__":
                print(v)
                print('----------'*20)
