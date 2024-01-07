import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
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
    memory = ConversationBufferMemory(memory_key='chat_history',  return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vector_store.as_retriever())
    return conversation_chain

def handle_user_question(prompt):
    pdfs = st.session_state.files
    if(len(pdfs) < 1):
        return st.warning('Please upload a PDF(s) and click on the Process button to begin', icon="⚠️")
    else:
        
        response = st.session_state.conversation_chain({"question": prompt})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                human = st.chat_message("user")
                # human.write(message.content, width=400, key='user_message')
                human.markdown(f"<div class='user-msg'>{message.content}</div>", unsafe_allow_html=True)
            else:
                 ai_assistant =  st.chat_message('AI')
                #  ai_assistant.markdown(message.content)
                 ai_assistant.markdown(f"<div class='ai-msg'>{message.content}</div>", unsafe_allow_html=True)
        

def markdown():
    return st.markdown(
        """
        <style>
            /* Style for user message */
            .user-msg {
                background-color: #000000;
                color: #ffffff;
                padding: 8px;
                border-radius: 8px;
                margin-bottom: 10px;
                width: fit-content;
                max-width: 70%;
            }
            /* Style for AI message */
            .ai-msg {
                background-color: #333333;
                color: #ffffff;
                padding: 8px;
                border-radius: 8px;
                margin-bottom: 10px;
                width: fit-content;
                max-width: 70%;
                align-self: flex-end;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title='Chat With Your PDF', page_icon=":books:")
    st.header("Chat With Multiple PDFs")
    markdown()

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None 
        # st.write('no conversation')
    # st.text_input("Ask questions about your documents")
        
    ai_assistant =  st.chat_message('AI')
    ai_assistant.write("Hello, I'm Bob's AI assistant, Please upload a pdf, click on the Process button and let's chat")

    prompt = st.chat_input('Ask something')
    if prompt:
        handle_user_question(prompt)

   

    # set a sidebar and put things in it 
    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload your pdf here and click on process", accept_multiple_files=True)
        st.session_state.files = pdfs
        if st.button("Process"):
            with st.spinner("Processing"):
                
                raw_text = get_pdf_text(pdfs)

                # split to chunks 
                text_chunks =  get_text_chunks(raw_text)

                # create a fiass vector store 
                vector_store = openai_vector_store(text_chunks)

                #create conversation chain 
                st.session_state.conversation_chain = create_conversation_chain(vector_store)
                st.write(vector_store)
                # st.success("Done processing ")
        


if __name__ == "__main__":
    main()