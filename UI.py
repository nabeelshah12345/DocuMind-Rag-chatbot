
import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


st.set_page_config(
    page_title="DocuMind",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 DocuMind")
st.caption("Upload a PDF and ask questions from its content.")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []



uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.session_state.vector_store is None:

        with st.spinner("Processing your PDF..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                pdf_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=10
            )

            chunks = splitter.split_documents(docs)

            # Create embeddings
            embedding = MistralAIEmbeddings()

            # Create Chroma database
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding
            )

            st.session_state.vector_store = vector_store

            # Remove temporary PDF
            os.remove(pdf_path)

        st.success("PDF processed successfully! You can now ask questions.")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.vector_store is not None:

    query = st.chat_input("Ask a question about your PDF...")

    if query:

        # Display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        # Retrieve documents
        docs = retriever.invoke(query)

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant.

                    Use only the provided context to answer the question.

                    If the answer is not in the context, say:
                    "I could not find the answer in the provided context."
                    """
                ),
                (
                    "user",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        # Create prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        # LLM
        llm = ChatMistralAI(
            model_name="ministral-8b-2512",
            temperature=0.1
        )

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:

    st.info("Upload a PDF to start chatting.")
