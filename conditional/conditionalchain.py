
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableBranch


load_dotenv()

model=ChatOpenAI()
parser1=StrOutputParser()
class Feedback(BaseModel):
    sentiment:Literal['Positive','Negative']=Field(description='Give the sentiment of the feedback')


# due to pydantic output will be consistent
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='classify the sentiment  of the following feedback text into positive or negative \n {feedback}\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain=prompt1|model|parser2

# res=classifier_chain.invoke({'feedback':'This is a terrile smartphone'}).sentiment


prompt2=PromptTemplate(
    template='Write  a appropriate respone to this positive feedback \n {feedback} ',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write  a appropriate respone to this negative feedback \n {feedback} ',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
#     (condition1,chain execute)
#     (condition2,chain execute )
    #    default chain
    
    (lambda x :x['sentiment']=='positive',prompt2|model|parser1)
    (lambda x :x['sentiment']=='negative',prompt3|model|parser1)
    RunnableLambda(lambda x:'could not find sentiment')
)



final_chain=classifier_chain|branch_chain
res=final_chain.invoke({'feedback':'This is a terrible phone'})
final_chain.get_graph().print_ascii()

print(res)