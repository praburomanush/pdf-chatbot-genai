import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceInstructEmbeddings,
)  # , HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS, Qdrant, Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline

# load environment variables
load_dotenv()
openai_api_key = "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"


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)


# get text chunks method
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_chunks = []
    position = 0
    # Iterate over the text until the entire text has been processed
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    return text_chunks


# get vector store method
def get_vectorstore(text_chunks):

    embeddings = OpenAIEmbeddings(
        openai_api_key="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"
    )
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    print(type(vector_store))

    return vector_store


# get conversation chain method
def get_conversation_chain(vectorstore):
    model_prams = {"temperature": 0.23, "max_length": 4096}
    llm = ChatOpenAI(
        openai_api_key="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"
    )

    # Alternatively, you can use a different language model, like Hugging Face's model
    # llm = HuggingFaceHub(repo_id="microsoft/phi-2", model_kwargs=model_prams)
    print("Creating conversation chain...")
    print("Conversation chain created")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Text vector retriever for context matching
        memory=memory,  # Memory buffer to store conversation history
    )


# get handler user input method
def handle_userinput(user_question):
    if st.session_state.conversation is not None:

        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
    else:
        st.write("Please upload PDFs and click process")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDF", page_icon=":PDF:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs :books:")
    if user_question := st.text_input("Ask a question about your documents:"):
        handle_userinput(user_question)

    # st.subheader("Model Parameters")

    # init sidebar
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click process", type="pdf", accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing PDFs"):
                process_files(pdf_docs, st)


def process_files(file_list, st):

    for file in file_list:
        file_extension = os.path.splitext(file.name)[1]
        file_name = os.path.splitext(file.name)[0]
        if file_extension == ".pdf":
            raw_text = get_pdf_text(file)
        elif file_extension == ".txt":
            with open(file, "r") as txt_file:
                raw_text = txt_file.read()

        elif file_extension == ".csv":
            with open(file, "r") as csv_file:
                raw_text = csv_file.read()

        else:
            raise Exception("File type not supported")

    print(raw_text)
    text_chunks = get_text_chunks(raw_text)
    print(f"Number of text chunks: {len(text_chunks)}")
    print("Creating vector store")
    vector_store = get_vectorstore(text_chunks)
    print("Vector store created")
    print("Creating conversation chain")
    st.session_state.conversation = get_conversation_chain(vector_store)
    print("Conversation chain created")


def get_file_text(file_path_list):
    raw_text = ""
    for file_path in file_path_list:
        file_extension = os.path.splitext(file_path)[1]
        file_name = os.path.splitext(file_path)[0]
        if file_extension == ".pdf":
            raw_text += get_pdf_text(file_path)
        elif file_extension == ".txt":
            with open(file_path, "r") as txt_file:
                raw_text += txt_file.read()

        elif file_extension == ".csv":
            with open(file_path, "r") as csv_file:
                raw_text += csv_file.read()

        else:
            raise Exception("File type not supported")

    return raw_text


if __name__ == "__main__":
    main()
