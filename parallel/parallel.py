from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableBranch


# we are using tiny llama

load_dotenv()

model1=ChatOpenAI()

llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1-1.B-Chat-v1.0"
    ,task="text-generation")

model2=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate simple and short notes from the following text {text} ',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template=' Generate 5 short question from the following  text {text} ',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='merge the provided notes and quiz into  a single document \n notes -> {notes} aand quiz -> {quiz} ',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser

})

chain3=prompt3|model1|parser

final_chain=parallel_chain|chain3

text="""

"""
res=final_chain.invoke({'text':text})
print(res)

# to see how your chain look
final_chain.get_graph().print_ascii()