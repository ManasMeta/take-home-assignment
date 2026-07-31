from langchain_core.documents import Document
from QA_bot import text_splitter, vector_database

print("Injecting sample document into Redis for testing...")

docs = [
    Document(
        page_content="The main topic of this document is artificial intelligence. The key findings are that AI improves RAG chatbot accuracy by 50%. The Watsonx platform provides state-of-the-art models for this purpose.", 
        metadata={"source": "dummy.pdf", "page": 1}
    )
]

chunks = text_splitter(docs)
vector_database(chunks)
print("Injection complete!")
