

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

embedding = MistralAIEmbeddings()

vector_store = Chroma(
    persist_directory="Chroma_db",
    embedding_function=embedding
)

# retriever

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {"k":5, "fetch_k": 10, "lambda_mult": 0.5 }
) 
llm = ChatMistralAI(model_name="ministral-8b-2512", temperature=0.1) 


# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are a helpful assistant.
         Use only provided context to answer the question.
         If answer is not in the context, say "I could not find the answer in the provided context." """ ),
        ("user", """
         Context: {context}
         Question: {question}
         """)
    ]
)

print("Rag System Created Successfully")
while True:
    query = input("Enter your question: ")
    if query.lower() == "0":
        break
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    final = prompt.invoke({"context": context, "question": query})
    
    response = llm.invoke(final)
    print("BOT : \n", response.content)

