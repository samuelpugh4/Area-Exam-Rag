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

# def highlightPDF(file, page_num, chunk):
#     doc=fitz.open(file)
#     page = doc[page_num]
    
#     #print("Searching for: Here")
#     rect1 = page.search_for(chunk[:10])[0].tl
#     #print("Found: ", len(rect1))
#     rect2 = page.search_for(chunk[-10:])[-1].br
#     #print("Found ", len(rect2))
#     page.add_highlight_annot(start=rect1, stop=rect2)
#     annotated_pdf = file.rstrip('.pdf') + '-Annotated.pdf'
#     doc.save(annotated_pdf)
#     return annotated_pdf

# def displayPDF(file,page_num):
#     # Opening file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="700" height="1000" type="application/pdf"></iframe>'
#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)

def updatePDF(file,page_num,chunk):
    print("Updating PDF to: ", file, " : ", page_num)
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')  
    st.session_state.pdf_to_show = base64_pdf
    st.session_state.page_to_show = page_num
    st.session_state.chunk_to_show = chunk

def displayPDF(placeholder):
    print("Calling displayPDF")
    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{st.session_state.pdf_to_show}#page={st.session_state.page_to_show}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    #placeholder = st.empty()
    placeholder.empty()
    #placeholder.write("Should display " + str(pdf_display))
    placeholder.markdown(pdf_display, unsafe_allow_html=True)

def generate_response(input_text):
    model = ChatOpenAI(openai_api_key=openai_api_key,model="gpt-3.5-turbo")
    st.info(model.invoke(input_text).content)
    #print("Session State: ", st.session_state)

def query_rag(query_text: str, k=3):

    print("In Query_Rag Function")
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    refs = st.session_state.references
    selected_refs = []
    for i, ref in enumerate(refs):
        if st.session_state[i]:
            selected_refs.append(ref)

    print("Searching across the following selected refs: ", selected_refs)

    #ref_retriever = db.as_retriever(search_kwargs={"k": int(k), "filter":{'source': {'$in': selected_refs}}})
    #print(ref_retriever)
    #print(type(ref_retriever))

    results = db.similarity_search_with_score(query_text, k=3, filter={'source': {'$in': selected_refs}})
    print("Results: ", results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print("Prompt for model: ", prompt)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")
    response_text = model.invoke(prompt).content
    st.info(response_text)

    return response_text, results

def display_sources(sources):
    print("In display sources function")
    #Create expander to show sources from RAG Query
    with st.expander(label='Sources Found for RAG Query', expanded=True):
        #Only do anything if > 0 sources
        if len(sources):
            #Make buttons for each source returned, update pdf when button clicked 
            for i, ref in enumerate(sources):
                src_id = ref[0].metadata['id']
                source = ref[0].metadata['source']
                page = ref[0].metadata['page']
                chunk = ref[0].page_content
                st.button(label=f'{src_id}', key=src_id, on_click=updatePDF, args=(source, page, chunk))

            #Save the source and page num to session state
            if 'pdf_to_show' not in st.session_state:
                #By default, show the first source
                with open(sources[0][0].metadata['source'], "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')  
                st.session_state.pdf_to_show = base64_pdf
            if 'page_to_show' not in st.session_state:
                page = sources[0][0].metadata['page']
                st.session_state.page_to_show = page

            if 'chunk_to_show' not in st.session_state:
                #print("sources[0][0]", sources[0][0])
                chunk = sources[0][0].page_content
                st.session_state.chunk_to_show = chunk
            
            #print("Now displaying the PDF")
            #print("Should display: ", st.session_state.pdf_to_show, " : ", st.session_state.page_to_show)
            #Now display the PDF
            chunk_placeholder = st.empty()
            chunk_placeholder.markdown(st.session_state.chunk_to_show)

            placeholder = st.empty()
            pdf_display = F'<iframe src="data:application/pdf;base64,{st.session_state.pdf_to_show}#page={st.session_state.page_to_show}" width="1000" height="1400" type="application/pdf"></iframe>'
            placeholder.markdown(pdf_display, unsafe_allow_html=True)

def selectAllReferences():
    for i, ref in enumerate(st.session_state.references):
        st.session_state[i] = True

def deselectAllReferences():
    for i, ref in enumerate(st.session_state.references):
        st.session_state[i] = False

### Checkboxes to select which references to include in Rag Search
#Get Database
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Create a variable to store refrences (From /data)
if not 'references' in st.session_state:
    st.session_state.references = []

#Get list of unique sources
for x in range(len(db.get()["ids"])):
    doc = db.get()["metadatas"][x]
    source = doc["source"]
    if source not in st.session_state.references:
        st.session_state.references.append(source)

#Display the contents of references as checkboxes
with st.expander(label='Selected Desired Sources for RAG Search', expanded=True):
    for i, ref in enumerate(st.session_state.references):
        st.checkbox(label=f'{ref}', key=i)
    col1, col2, _col3 = st.columns([.25,.25,.5])
    with col1:
        st.button('Select All', on_click=selectAllReferences)
    with col2:
        st.button('Deselect All', on_click=deselectAllReferences)

### Text Box for Query Submission
#Form for text box submission        
with st.form('my_form'):
    text = st.text_area('Enter text:', 'what is the coherence model and tangentiality model? Can ELMo embeddings differentiate patients from healthy people?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')

    if submitted and openai_api_key.startswith('sk-'):
        print("Querying rag with text: ", text)
        resp, sources = query_rag(text, k=3)
        st.session_state.response = resp
        st.session_state.returned_sources = sources

        #When you submit a new query, clear the last pdf / page that you were showing.
        #Will be automatically populated by new results
        if 'pdf_to_show' in st.session_state:
            del st.session_state['pdf_to_show']
        if 'page_to_show' in st.session_state:
            del st.session_state['page_to_show']
        if 'chunk_to_show' in st.session_state:
            del st.session_state['chunk_to_show']

if 'response' not in st.session_state:
    st.session_state.response = ''
with st.expander(label='Response', expanded=True):
    st.markdown(st.session_state.response)

if 'returned_sources' not in st.session_state:
    st.session_state['returned_sources'] = []
    
print("About to enter display sources function from main script")
display_sources(st.session_state.returned_sources)
