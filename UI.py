import streamlit as st
import tempfile
import os
import hashlib

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="DocuMind",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 DocuMind")
st.caption("Upload a PDF and ask questions from its content.")

load_dotenv()

try:
    MISTRAL_API_KEY = st.secrets.get(
        "MISTRAL_API_KEY",
        os.getenv("MISTRAL_API_KEY")
    )
except Exception:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY not found. "
        "Add it to your .env file locally or Streamlit secrets when deployed."
    )
    st.stop()



if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# Clear Chat
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Create unique hash for uploaded PDF
    current_file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    if current_file_hash != st.session_state.file_hash:

        with st.spinner("Processing your PDF..."):

            pdf_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )
                    pdf_path = temp_file.name


                loader = PyPDFLoader(pdf_path)

                docs = loader.load()

                if not docs:
                    st.error(
                        "Could not extract any pages from this PDF."
                    )
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(docs)

                if not chunks:
                    st.error(
                        "No text chunks were created from this PDF."
                    )
                    st.stop()

                chunks = [
                    chunk
                    for chunk in chunks
                    if chunk.page_content.strip()
                ]

                if not chunks:
                    st.error(
                        "The PDF does not contain readable text. "
                        "It may be an image/scanned PDF."
                    )
                    st.stop()

                embedding = MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=MISTRAL_API_KEY
                )

                test_embedding = embedding.embed_query(
                    "This is a test sentence."
                )

                if not test_embedding:
                    st.error(
                        "Mistral returned an empty embedding."
                    )
                    st.stop()

                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding
                )

                st.session_state.vector_store = vector_store

                # Save current PDF hash
                st.session_state.file_hash = current_file_hash

                # Clear previous conversation
                st.session_state.messages = []
                st.success("PDF processed successfully!")

            except Exception as e:

                st.error("Error while processing PDF:\n\n{str(e)}")

                st.session_state.vector_store = None

            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


if st.session_state.vector_store is not None:

    query = st.chat_input(
        "Ask a question about your PDF..."
    )
    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        retriever = (
            st.session_state.vector_store
            .as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
        )

        retrieved_docs = retriever.invoke(query)
        if retrieved_docs:

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in retrieved_docs
                ]
            )
        else:
            context = "No relevant context was found."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful assistant.

Answer the question using only the provided context.

If the answer is not present in the context, say:

"I could not find the answer in the provided context."
"""
                ),
                (
                    "user",
                    """
Context:{context}

Question:{question}
"""
                )
            ]
        )
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        llm = ChatMistralAI(
            model="ministral-8b-2512",
            temperature=0.1,
            api_key=MISTRAL_API_KEY
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(
                    final_prompt
                )
                answer = response.content
                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
else:
    st.info(
        "Upload a PDF to start chatting."
    )