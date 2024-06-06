from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


# define sample Tool
def multiply(input: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return input * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

import sys 
import os
sys.path.append(os.path.abspath("/data2/home/sahilkamble/llama3"))
from utils import *
llm, tokn = get_llama3(adapter_path="/data2/home/sahilkamble/llama3/models/Meta-Llama-3-8B-Instruct_ft_mhop")

cust_llm = LlamaIndexCustomLLM(llm, tokn, 1024)
agent = ReActAgent.from_tools([multiply_tool], llm=cust_llm, verbose=True)
agent.chat("What is 2*4")