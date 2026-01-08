# RAG Complaint Chatbot for Financial Services

A Retrieval-Augmented Generation (RAG) powered chatbot that transforms customer complaint data into actionable insights for CrediTrust Financial. This project enables product managers, support teams, and compliance officers to quickly understand customer pain points across financial products.

## 🎯 Business Objective

CrediTrust Financial receives thousands of customer complaints monthly across multiple channels. This AI tool helps internal stakeholders:

- **Decrease analysis time**: Identify complaint trends from days to minutes
- **Empower non-technical teams**: Get answers without data analysts
- **Proactive problem-solving**: Shift from reactive to proactive issue identification

## 📋 Project Overview

This project implements a complete RAG pipeline for analyzing customer complaints from the Consumer Financial Protection Bureau (CFPB) dataset, focusing on:

- **Credit Cards**
- **Personal Loans**
- **Savings Accounts**
- **Money Transfers**

## 🏗️ Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline
├── .vscode/
│   └── settings.json              # VS Code settings
├── data/
│   ├── raw/                       # Raw CFPB dataset (not in repo)
│   └── processed/                 # Cleaned and filtered data
├── vector_store/                  # Persisted vector databases
│   ├── chromadb/                  # ChromaDB vector store
│   └── faiss/                     # FAISS vector store (optional)
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb  # Task 1 notebook
│   └── README.md
├── src/
│   ├── data_processing.py         # Task 1: EDA and preprocessing
│   ├── embedding_pipeline.py      # Task 2: Chunking and embedding
│   └── rag_pipeline.py            # Task 3: RAG core logic
├── tests/
│   ├── test_data_processing.py
│   └── test_rag_pipeline.py
├── app.py                         # Task 4: Gradio interface
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- 8GB+ RAM (for embedding generation)
- CFPB complaint dataset (download from [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-complaint-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CFPB dataset**
   - Download the full CFPB complaint dataset
   - Place it in `data/raw/complaints.csv` (or update path in code)

## 📚 Task Implementation

### Task 1: Exploratory Data Analysis and Data Preprocessing

**Objective**: Understand and prepare complaint data for RAG pipeline.

**Implementation**: `src/data_processing.py`

**Usage**:
```bash
python src/data_processing.py
```

**Features**:
- Loads and analyzes CFPB complaint dataset
- Filters for target products (Credit Cards, Personal Loans, Savings Accounts, Money Transfers)
- Removes empty narratives
- Cleans text (lowercasing, removing boilerplate)
- Generates EDA visualizations
- Saves cleaned data to `data/processed/filtered_complaints.csv`

**Notebook**: `notebooks/task1_eda_preprocessing.ipynb`

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

**Objective**: Convert cleaned text into searchable vector format.

**Implementation**: `src/embedding_pipeline.py`

**Usage**:
```bash
python src/embedding_pipeline.py
```

**Features**:
- Creates stratified sample (10K-15K complaints) with proportional product representation
- Implements text chunking using LangChain's RecursiveCharacterTextSplitter
  - Chunk size: 500 characters
  - Chunk overlap: 50 characters
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Creates vector store using ChromaDB (or FAISS)
- Stores metadata (product, issue, complaint ID, etc.) with each chunk

**Configuration**:
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions, ~80MB)
- Vector store: ChromaDB (default) or FAISS
- Sample size: 12,000 complaints (configurable)

### Task 3: Building the RAG Core Logic and Evaluation

**Objective**: Build retrieval and generation pipeline with evaluation.

**Implementation**: `src/rag_pipeline.py`

**Usage**:
```bash
python src/rag_pipeline.py
```

**Features**:
- Loads pre-built vector store (or uses Task 2 output)
- Implements semantic search retriever (top-k=5)
- Prompt engineering for financial analyst assistant
- LLM integration (HuggingFace models)
- Evaluation framework with test questions

**Prompt Template**:
```
You are a financial analyst assistant for CrediTrust Financial. 
Your task is to answer questions about customer complaints based on 
the provided context from real complaint narratives.

Context: {context}
Question: {question}
Answer:
```

**Evaluation**:
- Test questions covering different products and issues
- Quality scoring (1-5 scale)
- Source verification and analysis

### Task 4: Creating an Interactive Chat Interface

**Objective**: Build user-friendly web interface for non-technical users.

**Implementation**: `app.py`

**Usage**:
```bash
python app.py
```

**Features**:
- Gradio-based web interface
- Natural language question input
- Real-time answer generation
- **Source display**: Shows retrieved complaint excerpts with metadata
- Clear chat functionality
- Responsive design

**Access**: Open browser to `http://localhost:7860`

## 🔧 Configuration

### Vector Store Options

**ChromaDB** (default):
- Persistent storage
- Metadata filtering
- Easy to use

**FAISS**:
- Faster search for large datasets
- Requires manual metadata management

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight (80MB)
- Fast inference
- Good quality for semantic search

Alternative models can be specified in `EmbeddingPipeline` initialization.

### LLM Models

Default: HuggingFace text generation models
- Can be configured in `RAGPipeline`
- Supports local and cloud-based models

## 📊 Data Requirements

### Input Data

1. **Full CFPB Dataset** (`data/raw/complaints.csv`)
   - Complete complaint dataset for Task 1
   - Required columns: Product, Consumer complaint narrative, Issue, etc.

2. **Pre-built Embeddings** (optional, `vector_store/complaint_embeddings.parquet`)
   - For Tasks 3-4, use pre-built embeddings if available
   - Contains ~1.37M chunks from 464K complaints

### Output Data

- `data/processed/filtered_complaints.csv`: Cleaned and filtered dataset
- `vector_store/chromadb/`: ChromaDB vector store
- `vector_store/faiss/`: FAISS index and metadata (if used)

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 📈 Evaluation

The RAG pipeline includes an evaluation framework:

1. **Test Questions**: 5-10 representative questions
2. **Quality Metrics**: Manual scoring (1-5 scale)
3. **Source Verification**: Check retrieved chunks for relevance
4. **Analysis**: Identify strengths and areas for improvement

Example evaluation questions:
- "Why are people unhappy with Credit Cards?"
- "What are the main issues with Personal Loans?"
- "What problems do customers face with Money Transfers?"

## 🎨 UI Features

The Gradio interface includes:

- **Chat Interface**: Natural conversation flow
- **Source Display**: Shows retrieved complaint excerpts
- **Product Filtering**: Metadata shows product, issue, complaint ID
- **Clear Functionality**: Reset conversation
- **Status Indicators**: System health monitoring

## 📝 Key Technical Decisions

1. **Chunking Strategy**: 500 characters with 50 overlap balances context and granularity
2. **Embedding Model**: all-MiniLM-L6-v2 chosen for speed/quality balance
3. **Vector Store**: ChromaDB for ease of use and metadata support
4. **Retrieval**: Top-5 chunks provide sufficient context without overwhelming LLM
5. **UI Framework**: Gradio for rapid development and deployment

## 🔮 Future Improvements

- [ ] Implement response streaming for better UX
- [ ] Add product-specific filtering in UI
- [ ] Integrate with cloud LLM APIs (OpenAI, Anthropic)
- [ ] Add automated evaluation metrics (BLEU, ROUGE, etc.)
- [ ] Implement conversation memory/history
- [ ] Add export functionality for reports
- [ ] Deploy as production service

## 🤝 Contributing

This project follows standard Python development practices:

1. Create feature branch
2. Write tests
3. Ensure all tests pass
4. Submit pull request

## 📄 License

This project is part of the 10 Academy AI Mastery program.

## 🙏 Acknowledgments

- Consumer Financial Protection Bureau for the complaint dataset
- 10 Academy for the challenge framework
- HuggingFace for open-source models and tools

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built for CrediTrust Financial | 10 Academy Week 7 Challenge**



