import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st
from langchain.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.vectorstores.base import VectorStore
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import streamlit as st
from langchain import LLMChain, OpenAI, PromptTemplate
import os
from langchain.chains import RetrievalQA, SimpleSequentialChain
import faiss

if st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    os.environ["BROWSERLESS_API_KEY"] = st.secrets['BROWSERLESS_API_KEY']
    os.environ["SERP_API_KEY"] = st.secrets['SERP_API_KEY']
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


embeddings_model = HuggingFaceEmbeddings()
# Initialize the vectorstore as empty

embedding_size = 768
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


loader = DirectoryLoader("customer_data", loader_cls=TextLoader)
# loader = TextLoader("customer_data/comp_complaints.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
docsearch = FAISS.from_documents(docs, embeddings_model)


customer_complaints = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0), chain_type="refine", retriever=docsearch.as_retriever()
)
