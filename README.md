# 🧠 ChaiCode RAG Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** system built using **LangChain**, **Google Gemini Pro**, and **Qdrant** to answer programming and all sorts of questions directly from the [ChaiCode Docs](https://docs.chaicode.com). It is designed to act as your intelligent programming tutor — answering your questions with clarity, depth, and direct source references.

This technology can be used to build RAG system on any website and webapp.

## 📖 What It Does

This project allows you to **ask any question** related to the content of **ChaiCode Docs** (HTML, Git, C, SQL, Django, DevOps), and the system will:

1. **Understand and expand your query contextually**  
2. **Search across topic-specific document collections** using semantic similarity  
3. **Fuse and rerank relevant documents** using Reciprocal Rank Fusion  
4. **Generate a detailed and educational answer** using Google Gemini  
5. **Provide clickable source URLs** from the ChaiCode documentation used to generate the answer

This means you don’t just get an answer — you get the **why**, the **how**, and the **where** it came from. Ideal for learners who value both clarity and traceability.

## 🚀 Features

🔗 Web-based loader for ChaiCode Docs URLs

❓ Parallel Query Expansion for broader semantic coverage

🔍 Routing of data sources across multiple topic-specific collections

🔍 Multi-collection Qdrant retrieval

🔁 Reciprocal Rank Fusion for optimal cross-collection document ranking

🧠 Gemini-powered LLM answers

🌐 URL references for source transparency

📚 Modular and extensible


## 🧱 Tech Stack

- **Python**
- [LangChain](https://www.langchain.com/)
- [Google Gemini Pro (1.5 Flash)](https://ai.google.dev/)
- [Qdrant Vector Store](https://qdrant.tech/)
- **Google Generative AI Embeddings**
- **ChaiCode Documentation**
- [Docker]

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chaicode-rag-assistant.git
cd chaicode-rag-assistant
2. Install Dependencies
Ensure you're using Python 3.9+:

bash
Copy
Edit
pip install -r requirements.txt
3. Set Up Environment Variables
Create a .env file with:

env
Copy
Edit
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
GOOGLE_API_KEY=your_google_api_key
Or enter keys when prompted.

4. Run the App
bash
Copy
Edit
python main.py
