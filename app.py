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
import pickle 
import os

# load env for custom api keys
load_dotenv()

# receives any number of pdfs, read each one of them,  extract the text contents and return it
def get_pdf_text(pdfs):
    text = ''
    
    for each_pdf in pdfs:
        pdf_reader = PdfReader(each_pdf)
        
        for each_pdf_doc in pdf_reader.pages:
            text+= each_pdf_doc.extract_text()
    return text


# The text contents we read from the  PDF may huge, more than what the we can pass to our AI model         
# This function splits text from documents into smaller chunks of 1000 characters, with 200 overlap
# we will later perform Text Embeddings with it
# I'm using CharacterTextSplitter here, you can use RecursiveTextSplitter
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len )
    return text_splitter.split_text(text)


# create a FAISS vector store, using openapi embeddings
# The FAISS vector store is in-memory vector store. 
# you can use chroma/pincone to store your vectors 
def openai_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# the above function uses OpenAI's embeddings, which is paid for. 
# If you machine is powerful, you can use one from huggingface 
# create a vector store with hugging face instructor-xl model, -quiet slow
def huggingface_vector_store(texts):
    embedding = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    return FAISS.from_texts(texts, embedding)

# use langchain to create a conversation chain, with conversation history
def create_conversation_chain(vector_store):
    llm= ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',  return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vector_store.as_retriever())
    return conversation_chain

# Accept users input, display both the question/answer  on the chat window after it's processed
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
                    ai_assistant.markdown(f"<div class='ai-msg'>{message.content}</div>", unsafe_allow_html=True)
                    



# markdown to format the chat ui interface
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

# check if a pickle file of a uploaded PDF document already exists in memory, don't recreate it's embeddings 
def check_and_load_pickle_files(pdfs):
    pickle_data = {}

    for each_pdf in pdfs:
        pickle_file = each_pdf.replace('.pdf', '.pickle')

        # check if pickle version of this pdf exist 
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                pickle_data[each_pdf] = pickle.load(file)
    return pickle_data

# main function of application, set the page config, sidebar and chat section of the app
def main():
    st.set_page_config(page_title='Chat With Your PDF', page_icon=":books:")
    st.header("Chat With Multiple PDFs")
    markdown()

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None 
        
    ai_assistant =  st.chat_message('AI')
    ai_assistant.write("Hello, I'm Bob's AI assistant, Please upload a pdf, click on the Process button and let's chat")

    prompt = st.chat_input('Ask something')
    if prompt:
        handle_user_question(prompt)

   

    # set a sidebar and put things in it 
    with st.sidebar:
        st.subheader("Your documents")
    
        pdfs = st.file_uploader("Upload your pdf here and click on process", accept_multiple_files=True, type="pdf")

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
                # st.write(vector_store)
                st.success("AI bot is ready to answer your questions ")
        


if __name__ == "__main__":
    main()