import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from get_embedding_function import get_embedding_function
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    print("In Main of Query Data")
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    #query_rag(query_text)

    '''Debug Database'''
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # alldocs = db.get()
    # for d in alldocs[:5]:
    #     print(d,"\n---\n")

    print(db.get().keys())
    print(len(db.get()["ids"]))

    # Print the list of source files
    for x in range(5):
        print("\n\n\n", x,": ----\n\n\n")
        # print(db.get()["metadatas"][x])
        doc = db.get()["documents"][x]
        #source = doc["source"]
        print(doc)

def query_rag(query_text: str):
    # Prepare the DB.
    #print("In Main of Query Data")

    embedding_function = get_embedding_function()
    #print("Got embedding function")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #print("Got DB")

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    #print("Got Results")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt for model: ", prompt)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4")
    #model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    # for r in results:
    #     print(r,"\n-----\n")

    return response_text


if __name__ == "__main__":
    main()
