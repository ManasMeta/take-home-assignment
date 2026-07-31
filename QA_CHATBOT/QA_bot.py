from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from redis import Redis
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

# Set up Redis LLM Cache
try:
    redis_client = Redis.from_url("redis://localhost:6379")
    set_llm_cache(RedisCache(redis_=redis_client))
except Exception as e:
    print(f"Warning: Could not connect to Redis for caching: {e}")


# LLM Initialization

def get_llm():
    llm = ChatGroq(
        model="llama3-8b-8192", 
        temperature=0.5,
        max_tokens=256
    )
    return llm



# Document Loader

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document



# Text Splitter

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks



# Embedding Model

def get_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings



# Vector Database

def vector_database(chunks):
    embedding_model = get_embedding()
    redis_url = "redis://localhost:6379"
    index_name = "qa_chatbot_index"
    vectordb = RedisVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        redis_url=redis_url,
        index_name=index_name
    )  
    return vectordb



# Retriever

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


# Retrieval-based QA Chain

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True 
    )

    response = qa.invoke({"query": query})
    return response["result"]


# Gradio Interface
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here..."
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="📄 Groq RAG-based PDF Q&A Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will answer using the content of your uploaded document."
)


# Launch the App

if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
