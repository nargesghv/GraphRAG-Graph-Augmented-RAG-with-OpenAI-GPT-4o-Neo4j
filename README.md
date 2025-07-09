
# 🔍 GraphRAG: Graph-Augmented RAG with OpenAI GPT-4o + Neo4j

This project demonstrates how to build a Graph Retrieval-Augmented Generation (GraphRAG) system using:

- 🧠 **OpenAI GPT-4o** for LLM-based text understanding and Cypher query generation.
- 🧬 **Neo4j** as a knowledge graph backend.
- 🔗 **LangChain** for chaining LLMs, prompts, and database logic.
- 🧾 Few-shot prompt engineering for Cypher query & QA response generation.

---

## 🚀 Features

- ✅ Transform unstructured text into a **structured knowledge graph**.
- ✅ Store and visualize that graph in **Neo4j**.
- ✅ Query the graph with **natural language**, powered by `gpt-4o`.
- ✅ Customize graph schemas, Cypher logic, and prompt behavior.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install langchain langchain-core langchain-openai langchain-experimental langchain-neo4j openai neo4j python-dotenv jupyter
```

---

## 🔐 .env Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here

NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password4
```

---

## 🧪 Running the Project

### 1. Start Neo4j (e.g. via Docker or Podman)

```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password4 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.15
```

### 2. Launch the Notebook

```bash
jupyter notebook
```

> Load the notebook/script and run each cell to build, load, and query your knowledge graph.

---

## 📊 Architecture

```mermaid
graph TD;
    A[Unstructured Text] --> B[LLM Graph Extractor (GPT-4o)];
    B --> C[LangChain GraphTransformer];
    C --> D[Neo4j Knowledge Graph];
    E[Natural Language Question] --> F[Cypher Prompt Generator];
    F --> G[LangChain QA Chain];
    G --> D;
    D --> H[Cypher Result];
    H --> I[LLM Answer Formatter];
    I --> J[Final Answer]
```

---

## 🧠 Example Questions

- “What is John's title?”
- “Who does Jane collaborate with?”
- “What group is Sharon part of?”

---

## 🛠️ Folder Structure

```
graphrag/
├── .env
├── requirements.txt
├── README.md
├── main.py  ← main code
```

---

## 📚 References

- [Neo4j Docs](https://neo4j.com/docs/)
- [LangChain](https://docs.langchain.com/)
- [OpenAI GPT-4o](https://platform.openai.com/)
- [GraphRAG](https://arxiv.org/abs/2306.11660)

---

## ✅ License

MIT License © 2025
