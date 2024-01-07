import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

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
            
# function to split text from documents into smaller chunks of 1000 characters, with 200 overlap
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len )
    return text_splitter.split_text(text)


def openai_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def huggingface_vector_store(texts):
    embedding = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    return FAISS.from_texts(texts, embedding)

def create_conversation_chain(vector_store):
    llm= ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='history', ai_prefix='AI', return_messages=True, human_prefix="Human")
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, memory, retriever=vector_store.as_retriever())
    return conversation_chain


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

                # split to chunks 
                text_chunks =  get_text_chunks(raw_text)

                # create a fiass vector store 
                vector_store = openai_vector_store(text_chunks)

                #create conversation chain 
                conversation_chain = create_conversation_chain(vector_store)
                st.write(vector_store)
                # st.success("Done processing ")
        


if __name__ == "__main__":
    main()