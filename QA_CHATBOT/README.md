# 🧠 Watsonx.ai RAG-based PDF Q&A Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built with:
- **IBM Watsonx.ai** (LLM & embeddings)
- **LangChain**
- **Gradio** for the UI

## 🚀 Features
- Upload a PDF and ask any question.
- Uses Watsonx.ai for context-aware answers.
- Simple Gradio interface.

## 🛠️ Setup
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv myenv
myenv\Scripts\activate  # or source myenv/bin/activate
pip install -r requirements.txt
python qabot.py



## 🧩 Tech Stack

| Component             | Technology / Model                                 |
|-----------------------|--------------------------------------------------|
| **Frontend/UI**       | Gradio                                           |
| **Large Language Model** | IBM Watsonx.ai Mixtral Model (`mistralai/mixtral-8x7b-instruct-v01`) |
| **Embeddings Model**  | IBM Watsonx.ai Slate Model (`ibm/slate-125m-english-rtrvr`)          |
| **Retrieval & Orchestration** | LangChain, LangChain-IBM, LangChain-Community          |
| **Vector Database**   | ChromaDB                                         |
| **Document Loader**   | PyPDF                                           |
| **Python Version**    | 3.11+                                           |

---

## 📸 Demo

Here’s a screenshot of the chatbot in action:

![PDF Chatbot Screenshot](assets/Screenshot.png)

"# chatwithdocs" 
