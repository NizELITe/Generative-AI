from langchain_openai import ChatopenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outPut_parsers import StrOutputParser 
from langchain_community.llms import Ollama 
from langchain_community.llms import Ollama 

import os 
from dotenv import load_dotenv

load_dotenv()


#os.environ["OPENAI_API_KEY"] = os.get_env("OPENAI_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environment["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Prompt Template
prompt=ChatPromptTemplate.from_messages(
[
    {"system","hey there iam chat gee pee taa"}
    {"user","Question:{question}"}
    
]
)

#ollama llama 2 (local lama 2)

llm =  ChatopenAI(model="llama2")
OutputParser = StrOutputParser
chain = prompt|llm|OutputParser


input_text = "give your input here"

output = chain.invoke({'question':input_text})

