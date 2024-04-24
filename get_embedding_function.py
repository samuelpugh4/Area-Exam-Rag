
from langchain_openai import OpenAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import os

def get_embedding_function():

    #print("In get embedding function")

    # openai_key =  os.environ.get('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings()
    return embeddings
