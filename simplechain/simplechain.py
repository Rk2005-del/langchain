from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt=PromptTemplate(
    template='Generate 5 interesting fact about {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()

# output in string format 
parser=StrOutputParser()

chain=prompt|model|parser
res=chain.invoke({'topic':'Ai'})
print(res)