import streamlit as st

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from get_embedding_function import get_embedding_function
import os
import base64
import fitz  # import package PyMuPDF


from streamlit_pdf_viewer import pdf_viewer


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

#st.title('Area Exam RAG Application')
openai_api_key=OPENAI_API_KEY

def highlightPDF(file, page_num, chunk):
    doc=fitz.open(file)
    page = doc[page_num]
    
    #print("Searching for: Here")
    rect1 = page.search_for(chunk[:10])[0].tl
    #print("Found: ", len(rect1))
    rect2 = page.search_for(chunk[-10:])[-1].br
    #print("Found ", len(rect2))
    page.add_highlight_annot(start=rect1, stop=rect2)
    annotated_pdf = file.rstrip('.pdf') + '-Annotated.pdf'
    doc.save(annotated_pdf)
    return annotated_pdf

def displayPDF(file,page_num):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def generate_response(input_text):
    model = ChatOpenAI(openai_api_key=openai_api_key,model="gpt-3.5-turbo")
    st.info(model.invoke(input_text).content)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Name an animal')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
        test_text = '''Here, we present the first benchmark
comparison of previously proposed coherence
models for detecting symptoms of schizophre-
nia and evaluate their performance on a new
dataset of recorded interviews between sub-
jects and clinicians.'''
        annotated_pdf = highlightPDF('data/Iter_2018.pdf',0,test_text)
        displayPDF(annotated_pdf, 0)
