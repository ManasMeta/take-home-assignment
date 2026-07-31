import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)

# We import the LLM and Embedding models from your existing project to use as Judges
from QA_bot import get_llm, get_embedding

def run_evaluation():
    print("Starting Evaluation Pipeline...")
    
    eval_data = {
        "question": [
            "What is the main topic of the document?",
            "What are the key findings?"
        ],
        "ground_truth": [
            "The main topic is artificial intelligence.",
            "The key findings are that AI improves RAG chatbot accuracy by 50%."
        ]
    }
    
    answers = []
    contexts = []
    
    print("Generating answers and retrieving contexts for evaluation...")
    
    # We will query your existing Redis vector store
    from langchain_community.vectorstores.redis import Redis as RedisVectorStore
    from langchain.chains import RetrievalQA
    
    try:
        embedding_model = get_embedding()
        vectordb = RedisVectorStore(
            redis_url="redis://localhost:6379",
            index_name="qa_chatbot_index",
            embedding=embedding_model
        )
        retriever_obj = vectordb.as_retriever()
        
        llm = get_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True 
        )

        for q in eval_data["question"]:
            print(f"  -> Asking: {q}")
            res = qa_chain.invoke({"query": q})
            answers.append(res["result"])
            # Extract text from source documents for context
            ctx = [doc.page_content for doc in res["source_documents"]]
            contexts.append(ctx)
            
    except Exception as e:
        print(f"❌ Error connecting to Redis or running QA Chain: {e}")
        print("Please ensure your QA_bot Gradio app has been run and a PDF was uploaded.")
        return

    # 2. Prepare Data for Ragas
    data = {
        "question": eval_data["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": eval_data["ground_truth"]
    }
    
    dataset = Dataset.from_dict(data)
    
    # 3. Setup Judge Models
    judge_llm = get_llm()
    judge_embeddings = get_embedding()
    
    print("\nEvaluating with Ragas (This might take a moment)...")
    
    try:
        # 4. Run Evaluation
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevance,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
        
        # 5. Output Results
        print("\nEvaluation Complete!")
        df = result.to_pandas()
        
        print("\n=== Final Scores ===")
        # Print the relevant metric columns
        cols_to_print = ['question', 'context_precision', 'faithfulness', 'answer_relevance', 'context_recall']
        print(df[cols_to_print].to_string(index=False))
        
        # Save to CSV
        df.to_csv("ragas_evaluation_results.csv", index=False)
        print("\nSaved full detailed results to ragas_evaluation_results.csv")
        
    except Exception as e:
         print(f"Error during Ragas evaluation: {e}")

if __name__ == "__main__":
    print("WARNING: Please edit 'eval_data' in this script with real questions and ground_truths for accurate results.")
    run_evaluation()
