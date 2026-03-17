# Euro-Laws-Agent-Helper

### Overview
Euro Laws Assistant is a RAG-based agent that answers questions about European legislation using EUR-Lex documents. It indexes ~ 140k laws (~700k chunks) and uses hybrid search (BM25 + vector search) with SentenceTransformers embeddings stored in Weaviate for retrieval. Built with LangGraph and LangChain, it leverages GPT-4o for reasoning and GPT-4.1 for conversational responses.
  

### Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-4B8BBE)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-FF6F00)
![Weaviate](https://img.shields.io/badge/Weaviate-00C7B7?logo=weaviate&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

### Repository Structure
```
Euro-Laws-Agent-Helper  
  ├── app/ → Web application layer (API + UI)
  │ ├── static/ → CSS, JS, images
  │ ├── templates/ → HTML templates
  │ └── main.py → Application entry point
  │
  ├── src/ → Core agent architecture
  │ ├── chains/ → LangChain pipelines
  │ ├── connection/ → External service connections (Weaviate, OpenAI)
  │ ├── core/ → Core logic and utilities
  │ ├── generation/ → LLM response generation
  │ ├── graph/ → LangGraph agent workflow
  │ ├── prompts/ → Prompt templates
  │ ├── retrieval/ → Vector search + RAG retrieval logic
  │ ├── sentencetransformers/ → Embedding model
  │ └── validation_layer/ → Output validation and guardrails
  │  
  ├── dataset/ → Legal documents used for indexing
  ├── inference/ → Query testing and inference scripts
  ├── notebooks/ → Experiments, indexing, and evaluation
  ├── docker/ → Docker configuration
  │
  ├── .env.example → Environment variable template
  ├── .gitignore
  └── README.md
  └── requirements.txt
```      

### 📊 Results
Example outputs from the Euro Laws Assistant answering legal queries using EUR-Lex documents:

| RAG Results |  
|--------------------------------|
| ![](Inference/agreements_companies.png) |
| ![](Inference/drug_chat.png) | 

**Performance Metrics:**
- Avg Retrieval Latency: ~1.0s  
- Avg Routing Token Completion: ~0.2s  
- Avg End-to-End Question Response Time: ~2.5s

### How to Run

1. **Clone the repository**

```bash
    git clone https://github.com/beshoyhakeem/Euro-Laws-Agent-Helper.git

```
2. **Move to project Directory**

```bash  
    cd Euro-Laws-Agent-Helper
```  


3. **Install dependencies**

```bash
    pip install -r requirements.txt

```

4. **Run the FastApi Page**
 
```bash
    uvicorn app.main:app --reload --port 8080

```


### Future Improvments
1. **Automated Data Pipeline – Implement an automated pipeline to scrape EUR-Lex data, chunk documents, generate embeddings, and update the Weaviate vector database.**
2. **Stronger Embeddings – Using a larger OpenAI embedding model to potentially improve retrieval quality.**
3. **Specialized Local LLM – Train or fine-tune a domain-specific LLM with a larger context window using SentenceTransformers for secure local deployment.**


### 👤 Author
**Beshoy Hakeem**  
[LinkedIn](https://www.linkedin.com/in/beshoy-fahmy-14a254359/)  
[GITHUB](https://github.com/beshoyhakeem)  
Email: beshoyashraf042@gmail.com