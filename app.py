import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

def get_pdf_text(pdfs):
    text = ''
    # loop through each pdf provided and set it to a variable pdf_reader
    for each_pdf in pdfs:
        pdf_reader = PdfReader(each_pdf)
        # loop through each single page for each of the pdf documents, extract the text and return it
        for each_pdf_doc in pdf_reader.pages:
            text+= each_pdf_doc.extract_text()
    return text
            

def main():
    st.set_page_config(page_title='Chat With Your PDF', page_icon=":books:")
    st.header("Chat With Multiple PDFs")
    st.text_input("Ask questions about your documents")

    # set a sidebar and put things in it 
    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload your pdf here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                raw_text = get_pdf_text(pdfs)
                st.write(raw_text)
                # st.success("Done processing ")
        


if __name__ == "__main__":
    main()