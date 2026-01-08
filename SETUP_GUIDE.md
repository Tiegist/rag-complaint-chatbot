# Setup Guide - RAG Complaint Chatbot

This guide will help you set up and run the complete RAG complaint chatbot project.

## Prerequisites

1. **Python 3.11+** installed
2. **8GB+ RAM** (for embedding generation)
3. **CFPB Complaint Dataset** - Download from:
   - https://www.consumerfinance.gov/data-research/consumer-complaints/
   - Or use the provided pre-built embeddings if available

## Step-by-Step Setup

### 1. Clone and Navigate

```bash
cd rag-complaint-chatbot
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```



### 4. Download Data

#### Option A: Full Dataset (for Task 1-2)
1. Download the CFPB complaint dataset
2. Place it in `data/raw/complaints.csv`
3. The file should be a CSV with columns like:
   - Product
   - Consumer complaint narrative
   - Issue
   - Sub-issue
   - Company
   - State
   - Date received

#### Option B: Pre-built Embeddings (for Task 3-4)
1. If you have `complaint_embeddings.parquet`, place it in `vector_store/`
2. This allows you to skip Task 2 and go directly to Task 3-4

### 5. Run Tasks

#### Task 1: EDA and Preprocessing

**Option 1: Using Python script**
```bash
python src/data_processing.py
```

**Option 2: Using Jupyter Notebook**
```bash
jupyter notebook notebooks/task1_eda_preprocessing.ipynb
```

**Option 3: Using helper script**
```bash
python run_pipeline.py --task 1
```

**Expected Output:**
- EDA visualizations in `data/processed/`
- Cleaned dataset: `data/processed/filtered_complaints.csv`

#### Task 2: Embedding Pipeline

**Using Python script:**
```bash
python src/embedding_pipeline.py
```

**Using helper script:**
```bash
python run_pipeline.py --task 2
```

**Expected Output:**
- Vector store in `vector_store/chromadb/` (or `vector_store/faiss/`)
- This may take 30-60 minutes depending on your hardware

**Configuration:**
- Sample size: 12,000 complaints (configurable in code)
- Chunk size: 500 characters
- Embedding model: all-MiniLM-L6-v2

#### Task 3: RAG Pipeline and Evaluation

**Using Python script:**
```bash
python src/rag_pipeline.py
```

**Using helper script:**
```bash
python run_pipeline.py --task 3
```

**Expected Output:**
- Evaluation results: `data/processed/evaluation_results.csv`
- Console output with test questions and answers

#### Task 4: Launch Chat Interface

**Using Python script:**
```bash
python app.py
```

**Using helper script:**
```bash
python run_pipeline.py --task 4
```

**Access:**
- Open browser to: `http://localhost:7860`
- The interface will be available locally

### 6. Run All Tasks (1-3)

To run tasks 1-3 in sequence:

```bash
python run_pipeline.py --all
```

**Note:** Task 4 (UI) must be run separately as it starts a web server.

## Troubleshooting

### Issue: "FileNotFoundError: data/raw/complaints.csv"

**Solution:**
- Download the CFPB dataset
- Place it in `data/raw/complaints.csv`
- Or update the path in the code

### Issue: "ChromaDB not found" or "FAISS index not found"

**Solution:**
- Run Task 2 first to create the vector store
- Or place `complaint_embeddings.parquet` in `vector_store/`

### Issue: Out of Memory during embedding

**Solution:**
- Reduce sample size in Task 2 (e.g., 5000 instead of 12000)
- Use pre-built embeddings if available
- Close other applications

### Issue: LLM model fails to load

**Solution:**
- The code includes fallback template-based responses
- For better results, ensure transformers library is properly installed
- Consider using cloud-based LLM APIs (OpenAI, Anthropic) for production

### Issue: Gradio interface doesn't start

**Solution:**
- Check if port 7860 is already in use
- Change port in `app.py`: `demo.launch(server_port=7861)`
- Ensure all dependencies are installed

## Project Structure After Setup

```
rag-complaint-chatbot/
├── data/
│   ├── raw/
│   │   └── complaints.csv          # Your downloaded dataset
│   └── processed/
│       ├── filtered_complaints.csv # Task 1 output
│       ├── evaluation_results.csv  # Task 3 output
│       └── *.png                   # EDA visualizations
├── vector_store/
│   ├── chromadb/                   # Task 2 output (ChromaDB)
│   └── complaint_embeddings.parquet # Optional pre-built embeddings
└── [other project files]
```

## Next Steps

1. **Review EDA Results**: Check `data/processed/` for visualizations
2. **Evaluate RAG**: Review `data/processed/evaluation_results.csv`
3. **Test UI**: Try different questions in the chat interface
4. **Customize**: Modify prompts, chunk sizes, or models as needed

## Getting Help

- Check the main `README.md` for detailed documentation
- Review code comments in `src/` modules
- Open GitHub issues for bugs or questions

## Performance Tips

1. **For faster embedding**: Use pre-built embeddings when available
2. **For better answers**: Increase `top_k` in RAG pipeline (default: 5)
3. **For production**: Consider using cloud LLM APIs
4. **For larger datasets**: Use FAISS instead of ChromaDB

---

**Happy coding! 🚀**



