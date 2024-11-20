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
                text = ""
         for name in pdfs_to_summarize:
                if name in st.session_state['pdf_texts']:
                    text += st.session_state['pdf_texts'][name]
                else:
                    st.warning(f"Warning: PDF '{name}' is not processed yet.")
            if not text.strip():
                st.error("Error: The selected PDFs contain no readable text.")
                return
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
                st.session_state['summary'] = summary
                st.write("Summary:", summary)
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
