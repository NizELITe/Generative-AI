from langchain_openai import ChatopenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outPut_parsers import StrOutputParser 

import os 
from dotenv import load_dotenv 

os.environ["OPENAI_API_KEY"] = os.get_env("OPENAI_API_KEY")
#Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environment["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Prompt Template
prompt=ChatPromptTemplate.from_messages(
[
    {"system","hey there iam chat gee pee taa"}
    {"user","Question:{question}"}
    
]
)

#openai llm 

llm =  ChatopenAI(model="gpt-3.5-turbo")
OutputParser = StrOutputParser
chain = prompt|llm|OutputParser


input_text = "give your input here"

output = chain.invoke({'question':input_text})



#to run we need to setup on smith langchain
