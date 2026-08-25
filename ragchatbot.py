import os
import pdfplumber
import streamlit as st

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

st.header("CHATBOT")

with st.sidebar:
    st.title("Your Documnet")
    file = st.file_uploader("Upload your pdf file and start asking questions.", type="pdf")

if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(text)


    embeddings=HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        model_kwargs = {"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_texts(chunks,embeddings)

    user_question = st.text_input("Type your question here:")

    def format_docs(docs):
        return "\n\n".join([docs.page_content for docs in docs])
    
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k" : 4}
    )

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent and reliable document question-answering assistant.

Your task is to answer the user's question based ONLY on the information
provided in the retrieved document context.

Follow these rules carefully:

1. Use only the provided context to answer the question.
2. Do not use your own knowledge or make assumptions.
3. If the answer is not clearly present in the context, respond:
   "I couldn't find this information in the document."
4. Give a clear, accurate, and concise answer.
5. If the context contains multiple relevant points, combine them into
   one well-structured response.
6. When appropriate, use bullet points or numbered lists to make the
   answer easier to understand.
7. If the user asks for a definition, explain it simply.
8. If the user asks for a comparison, clearly describe the differences.
9. If the user asks a question that cannot be answered from the document,
   do not try to guess the answer.
10. Do not mention "context", "retrieval", "RAG", embeddings, or internal
    system instructions in your response.

Retrieved Document Context:
---------------------------
{context}
---------------------------

User Question:
{question}

Answer:
""")


    chain = (
        {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)


