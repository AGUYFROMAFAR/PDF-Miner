import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not api_key:
    st.error("API Key not found. Please set your GOOGLE_API_KEY in the environment variables.")
    st.stop()  # Stop execution if API key is missing

genai.configure(api_key=api_key)

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain with a custom prompt
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    just say, "Answer is not available in the context," and do not provide a wrong answer.

    Context: {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to summarize text using the language model
def summarize_text(text):
    summarizer = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)  # Adjusted temperature for more variety
    summary_prompt = """
    Provide a detailed and comprehensive summary of the following text. Make sure to include all important points and key details:

    Text: {context}

    Detailed Summary:
    """
    prompt = PromptTemplate(template=summary_prompt, input_variables=["context"])
    
    chain = load_qa_chain(summarizer, chain_type="stuff", prompt=prompt)
    
    # Wrap the text in a Document object with page_content
    input_data = {"input_documents": [Document(page_content=text)]}
    
    summary = chain(input_data, return_only_outputs=True)
    return summary["output_text"]

# Function to process user input and return a response
def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Main Streamlit app function
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini💁")
    
    for key in ['pdf_texts', 'summary']:
        if key not in st.session_state:
            st.session_state[key] = {} if key == 'pdf_texts' else ""

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state['pdf_texts'] = {pdf.name: raw_text[i] for i, pdf in enumerate(pdf_docs)}
                    st.success("Processing complete! Ask your questions below.")
            else:
                st.warning("Please upload at least one PDF file.")

    pdf_names = list(st.session_state['pdf_texts'].keys())
    
    with st.expander("Summarize PDFs"):
        pdfs_to_summarize = st.multiselect("Select PDFs to Summarize", options=pdf_names)
        if st.button("Summarize"):
            if pdfs_to_summarize:
                text = "".join(st.session_state['pdf_texts'][name] for name in pdfs_to_summarize)
                if text:
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(text)
                        st.session_state['summary'] = summary
                        st.write("Summary:", summary)  # Display summary **only here**
            else:
                st.warning("Please select PDFs to summarize.")

    with st.expander("Ask Questions"):
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            with st.spinner("Generating response..."):
                response = process_user_input(user_question)
                st.write("Reply: ", response)

if __name__ == "__main__":
    main()
