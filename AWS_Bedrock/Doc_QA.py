import json
import os
import sys
import boto3
import streamlit as st

## we are using Titan Embeddings Model to generate Embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

#Data Ingestion-libraries
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

#from langchain_community.document_loaders import PyPDFLoader

#vector Embeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

#For LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

#Bedrock clients
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-text-express-v1",client=bedrock)

##Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs

#vector Embeddings - vectors store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

#LLM model

def get_titan():
    llm=Bedrock(
        model_id = "amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs = {'maxTokenCount':512}
    )
    return llm

def get_llama3():
    llm=Bedrock(
        model_id = "meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs = {'max_gen_len':512}
    )
    return llm

#Prompt Template

prompt_template="""

Human: Use the following context to provide a concise answer to the question at the end 
use atleast summarize with 250 words with detailed explanation.
<context>
{context}
</context>

Question : {question}
Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

#Response
def get_response_llm(llm,vectorstore_faiss,query):
    qa=retrieval_qa.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity",search_kwargs={"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt":PROMPT}
    )

    answer = qa({"query":query})
    return answer['result']

#+++++++++++++++++++++++==========STREAMLIT=============+++++++++++++++++++++++++++++++++++++++



def main():
    st.set_page_config("Chat With PDF")
    st.header("Chat with PDF using AWS Bedrock 🤖🤖")

    user_question = st.text_input("ASk Question from PDF Files")

    with st.sidebar:
        st.title("Update_Create Vectors")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                #calling DataIngestion Function
                docs = data_ingestion()
                #calling vector Embeddings---> Saving in Local path
                get_vector_store(docs)
                st.success("Done")


    if st.button("Amazon Titan Output"):
        with st.spinner("Processing..."):
            #Loading Vectors from local path
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings)
            llm=get_titan()

            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success

if __name__=="__main__":
    main()








