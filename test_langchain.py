import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


from get_embedding_function import get_embedding_function
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PROMPT_TEMPLATE = """
Answer the question to the best of your ability: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text


    
    '''
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    #prompt = prompt_template.format( question=query_text)
    messages = [
    SystemMessage(
        content="You are a helpful assistant that will answer questions to the best of your ability. If you are unable to answer the question accurately, you should not attempt to answer it."
    ),
    HumanMessage(content=prompt_template.format(question=query_text)),
    ]
    #print("Prompt for model: ", prompt)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")
    response = model.invoke(messages)
    #print(response)
    response_content = response.content
    formatted_response = f"\nResponse: {response_content}"
    print(formatted_response)
    '''

if __name__ == "__main__":
    main()