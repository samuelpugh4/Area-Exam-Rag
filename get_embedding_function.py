from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import os

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    print("In get embedding function")
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # openai_key =  os.environ.get('OPENAI_API_KEY')
    # embeddings = embedding_functions.OpenAIEmbeddingFunction(
    #             api_key=openai_key,
    #             model_name="text-embedding-ada-002"
    #             #model_name="text-embedding-3-small"
    #         )

    #embeddings = embedding_functions.DefaultEmbeddingFunction()
    embeddings = OpenAIEmbeddings()
    return embeddings
