# """

# we use these things in separate file because we dont want to
# recreate these things again and again when we rerun our
# pyhton file

# 1 : Load pdf
# 2 : split into chunks
# 3 : create the embeddings
# 4 : store into chroma db

# """


from langchain_community.document_loaders import  PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

text_loader = PyPDFLoader(file_path="document loaders/GRU.pdf")  
docs = text_loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10 
)
chunks = splitter.split_documents(docs)

embedding = MistralAIEmbeddings(model="codestral-embed-2505")

total = len(chunks)  # total = 889

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="Chroma_db"
)

def batch_add_to_chroma(vector_store, chunks, batch_size=10):
   
    for i in range(0, total, batch_size):         # i = 0, 10, 20, 30, ... 880
        batch = chunks[i:i + batch_size]          # slice: chunks[0:10], chunks[10:20], ...
        vector_store.add_documents(batch)         # sirf ye 10 chunks Mistral ko bhejo


batch_add_to_chroma(vector_store, chunks, batch_size=10)
