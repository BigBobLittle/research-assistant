import streamlit as st 

def main():
    st.set_page_config(page_title='Chat With Your PDF', page_icon=":books:")
    st.header("Chat With Multiple PDFs")
    st.text_input("Ask questions about your documents")

    # set a sidebar and put things in it 
    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your pdf here and click on process")
        st.button("Process")
        


if __name__ == "__main__":
    main()