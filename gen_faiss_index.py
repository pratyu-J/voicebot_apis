# imports
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import UnstructuredURLLoader
import re, os, openai, glob
import os
import logging
from decouple import config
# from bs4 import BeautifulSoup as Soup
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate


# OpenAI
openai_key = config("OPENAI_API_KEY")  # Open AI API token
openai.api_key = openai_key
os.environ['OPENAI_API_KEY'] = openai_key
llm = ChatOpenAI(model_name="gpt-4-turbo")
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings()
chain = None


def get_response(chain, message):
    inputs = {"question": message}
    outputs = chain(inputs)
    response = outputs["answer"]
    print(response)
    print("chain", outputs)
    # source_metadata=get_source_metadata(outputs['source_documents'],response)
    message_response=response 
    return message_response

def get_index_directory(user_id,servicetype, usertype, question):
    basedir = "/app"
    print(servicetype)
    print(usertype)
    if(servicetype == "Status Amendment"):
        if(usertype == "Citizen"):
            print("here?????")
            index_dir1=os.path.join(basedir,f"Status_amendment")
            index_dir2=os.path.join(basedir,f"status_edit")
            print(index_dir1 + "\n" + index_dir2)
        if(usertype == "GCC"):
            index_dir1=os.path.join(basedir,f"Status_amendment")
            index_dir2=os.path.join(basedir,f"status_edit")
            print(index_dir1 + "\n" + index_dir2)
        if(usertype == "Foreign"):
            index_dir1=os.path.join(basedir,f"Status_amendment")
            index_dir2=os.path.join(basedir,f"status_edit")
            print(index_dir1 + "\n" + index_dir2)
    elif(servicetype == "Residency renewal"):
        if(usertype == "Citizen"):
            print("there?????")
            index_dir1=os.path.join(basedir,f"Residency_renewal_citizen_family_members")
            index_dir2=os.path.join(basedir,f"Residency_visa_renewal_citizen_gcc_family")
            print(index_dir1)
        if(usertype == "GCC"):
            index_dir1=os.path.join(basedir,f"Residency_renewal_gcc_family_members")
            index_dir2=os.path.join(basedir,f"Residency_visa_renewal_citizen_gcc_family")
            print(index_dir1)
        if(usertype == "Foreign"):
            index_dir1=os.path.join(basedir,f"Residency_renewal_foreigner_family_members")
            index_dir2=os.path.join(basedir,f"Residency_renewal_foreign_members")
            print(index_dir1)
    
    print("fetching vector")
    vectorstore=FAISS.load_local(index_dir1,embeddings, allow_dangerous_deserialization=True)
    vectorstore2=FAISS.load_local(index_dir2,embeddings,allow_dangerous_deserialization=True)
    
    print("Merging vectors")
    vectorstore.merge_from(vectorstore2)
    print("fetched vector##")
    print(f'{index_dir1} loaded')
    
    
    template = '''

    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer.
    First look at the data extracted from the pdf, if the answer was not found in the pdf data only then look at the data gathered from the URL.
    Try to give complete information on the question. Summarize if you have to. 
    Dont mention the source link.

    Use the context (delimited by <ctx></ctx>) and chat history (delimited by <hs></hs>) to see if any relevant information is there and use that to answer the query.
    <ctx>
    {context}
    context: You are a useful bot who can provide information about the products and services offered by the General Directorate of Residency and Foreigners Affairs - Dubai.


    </ctx>

    <hs>
    {history}
    </hs>

    Question: {question}
    Answer:
    '''
    prompt = PromptTemplate(
        template=template,
        input_variables=['history', 'context', 'question']
    )
    
    llm=ChatOpenAI(model_name='gpt-4-turbo')
    
    retriever=vectorstore.as_retriever() #(search_kwargs={'k':6})search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7}
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "memory": ConversationBufferWindowMemory(
                k=3,
                memory_key="history",
                input_key="question"),

        }
    )
    # return chain
    with get_openai_callback() as cb:
        d_response=chain.invoke({"query": question})
        print(d_response["result"])
        answer = d_response["result"]
        print("Sources:")
        sources_set = set()
        for source_document in d_response["source_documents"]:
            sources_set.add(source_document.metadata['source'])
        for source in sources_set:
            print("source= ", source)
        print("cb= ",cb)
        return answer + "\n \n" + source

# for voice bot
# Define user_memory globally to persist across function calls
user_memory = {}

def get_or_create_chain(user_id ,vectorstore):

    template = '''
    Based on the chat history and the provided question, determine if sufficient details are already available.
    Details are sufficient if you know whether it is residency renewal or status amendment, if the user is Dubai Citizen or GCC Citizen or a foreign national and whether the service is for them or a family member.
    Use the chat history to avoid asking redundant questions and only ask for the missing information.

    If additional information is still required to answer accurately:
    1. Determine if the query is about **Residency Renewal** or **Status Amendment**.
    2. Check if the user's classification (Dubai Citizen, GCC Citizen, Foreign National) has been mentioned.
    3. Check if the service is for the user or a **family member**.
    
    Remember:
    1. In the provided data, "citizen" refers to **Dubai Citizen**, "edit status" refers to **Status Amendment**, and "residency renewal" refers to **Residency Renewal**.
    2. If the answer is not found in the PDF data, ONLY THEN refer to the URL data.
    3. If you do not know the answer, say that you don’t know. Do not make up an answer.
    Once you have all the information required, give the process for the same.
    Don't mention words like chat history in the response.
    Keep the answers brief, in less than 1500 characters.
    Your response should always be in the same language as the question.
    <ctx>
    {context}
    </ctx>

    Chat History:
    {chat_history}

    Question: {question}
    '''

    # Define the PromptTemplate
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'chat_history', 'question']
    )
    print("before making user memory")
    # Check if memory already exists for the user
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True,
            output_key='answer'
        )
    print("after memory creation")
    # Create or reuse the chain with the existing memory
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=user_memory[user_id],
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True,
        return_generated_question=True,
        output_key="answer"
    )

def getVoiceResponse(question, user_id):
    basedir = "D:/gen_ai_bot/gdrfa/"
    # basedir = "/app"
    merged_index_dir = os.path.join(basedir, f"merged_index")
    vectorstore = FAISS.load_local(merged_index_dir, embeddings)

    # Get or create the chain
    print("before getorcreatechain")
    chain = get_or_create_chain(user_id ,vectorstore)
    print("before getorcreatechain")
    # Use the chain to generate a response
    with get_openai_callback() as cb:
        print("inside ipenai callback")
        d_response = chain.invoke({"question": question})
        print(d_response["answer"])
        answer = d_response["answer"]
        print("Sources:")
        sources_set = set()
        for source_document in d_response["source_documents"]:
            sources_set.add(source_document.metadata['source'])
        for source in sources_set:
            print("source= ", source)
        print("cb= ", cb)
        print("User Memory", user_memory)
        
        return answer
