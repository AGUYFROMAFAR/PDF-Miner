import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please set your GOOGLE_API_KEY in the environment variables.")
    st.stop()

genai.configure(api_key=api_key)

# Function to extract text from a PDF
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

# Function to summarize text using the model
def summarize_text(text):
    summarizer = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    summary_prompt = """
    Provide a detailed and comprehensive summary of the following text:
    
    Text: {context}
    
    Summary:
    """
    prompt = PromptTemplate(template=summary_prompt, input_variables=["context"])
    chain = load_qa_chain(summarizer, chain_type="stuff", prompt=prompt)
    summary = chain.run({"input_documents": [{"page_content": text}]})
    return summary

# Main app function
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini💁")

    if "processed_pdfs" not in st.session_state:
        st.session_state["processed_pdfs"] = set()

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    if st.button("Summarize PDFs"):
        if pdf_docs:
            for pdf in pdf_docs:
                if pdf.name in st.session_state["processed_pdfs"]:
                    st.info(f"PDF '{pdf.name}' has already been summarized.")
                else:
                    st.session_state["processed_pdfs"].add(pdf.name)
                    raw_text = get_pdf_text([pdf])
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    summary = summarize_text(raw_text)
                    st.write(f"Summary for '{pdf.name}':")
                    st.write(summary)
        else:
            st.warning("Please upload at least one PDF file.")

if _name_ == "_main_":
    main()
