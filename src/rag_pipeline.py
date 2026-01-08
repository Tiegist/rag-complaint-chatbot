"""
RAG (Retrieval-Augmented Generation) pipeline module.
Task 3: Build retrieval and generation pipeline using pre-built vector store.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
except ImportError:
    # Fallback for newer langchain versions
    try:
        from langchain_community.llms import HuggingFacePipeline
        from langchain.prompts import PromptTemplate
    except ImportError:
        # Simple template if langchain not available
        class PromptTemplate:
            def __init__(self, template, input_variables):
                self.template = template
                self.input_variables = input_variables
            def format(self, **kwargs):
                return self.template.format(**kwargs)
import warnings
warnings.filterwarnings("ignore")


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for complaint analysis."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "chromadb",
        vector_store_path: str = "vector_store",
        llm_model_name: Optional[str] = None,
        use_huggingface: bool = True,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model_name: Name of embedding model (must match Task 2)
            vector_store_type: "chromadb" or "faiss"
            vector_store_path: Path to vector store directory
            llm_model_name: Name of LLM model (if None, uses default)
            use_huggingface: Whether to use HuggingFace models
            top_k: Number of chunks to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store_type = vector_store_type
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded!")
        
        # Load vector store
        self._load_vector_store()
        
        # Initialize LLM
        self._initialize_llm(llm_model_name, use_huggingface)
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _load_vector_store(self):
        """Load the vector store (ChromaDB or FAISS)."""
        if self.vector_store_type == "chromadb":
            self._load_chromadb()
        else:
            self._load_faiss()
    
    def _load_chromadb(self):
        """Load ChromaDB vector store."""
        print(f"Loading ChromaDB from {self.vector_store_path}/chromadb...")
        
        # Try to load from pre-built embeddings parquet first
        parquet_path = Path(self.vector_store_path) / "complaint_embeddings.parquet"
        if parquet_path.exists():
            print("Loading from pre-built embeddings parquet...")
            self._load_from_parquet(parquet_path)
            return
        
        # Otherwise load from ChromaDB
        chroma_path = Path(self.vector_store_path) / "chromadb"
        if not chroma_path.exists():
            raise FileNotFoundError(f"ChromaDB not found at {chroma_path}")
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.chroma_collection = self.chroma_client.get_collection("complaint_chunks")
        print("ChromaDB loaded successfully!")
    
    def _load_faiss(self):
        """Load FAISS vector store."""
        print(f"Loading FAISS from {self.vector_store_path}/faiss...")
        
        faiss_path = Path(self.vector_store_path) / "faiss"
        index_path = faiss_path / "faiss.index"
        metadata_path = faiss_path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            self.faiss_metadata = pickle.load(f)
        
        print(f"FAISS loaded with {self.faiss_index.ntotal} vectors!")
    
    def _load_from_parquet(self, parquet_path: Path):
        """Load embeddings and metadata from pre-built parquet file."""
        print(f"Loading from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Extract embeddings and metadata
        if 'embedding' in df.columns:
            embeddings = np.array(df['embedding'].tolist())
        elif 'embeddings' in df.columns:
            embeddings = np.array(df['embeddings'].tolist())
        else:
            raise ValueError("No embedding column found in parquet file")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        metadata_cols = [col for col in df.columns if col != 'embedding' and col != 'embeddings']
        self.faiss_metadata = df[metadata_cols].to_dict('records')
        
        # Update vector store type to faiss for retrieval
        self.vector_store_type = "faiss"
        
        print(f"Loaded {len(self.faiss_metadata)} chunks from parquet!")
    
    def _initialize_llm(self, model_name: Optional[str], use_huggingface: bool):
        """Initialize the language model."""
        if not use_huggingface:
            # For production, you might use OpenAI, Anthropic, etc.
            print("Using HuggingFace models (default)")
            use_huggingface = True
        
        # Default to a small, fast model for local use
        if model_name is None:
            model_name = "microsoft/DialoGPT-medium"  # Lightweight option
            # Alternative: "mistralai/Mistral-7B-Instruct-v0.2" (requires more resources)
        
        print(f"Loading LLM: {model_name}...")
        try:
            # Try to use a text generation pipeline
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            print("LLM loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Using simple template-based responses instead.")
            self.llm = None
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the LLM."""
        template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context from real complaint narratives.

Use the following retrieved complaint excerpts to formulate your answer. Be specific and cite examples from the context when possible. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant chunks with metadata
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        if self.vector_store_type == "chromadb":
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.top_k
            )
            
            # Format results
            retrieved_chunks = []
            for i in range(len(results['ids'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk)
        else:
            # Query FAISS
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.faiss_index.search(query_embedding, self.top_k)
            
            # Format results
            retrieved_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.faiss_metadata):
                    chunk = {
                        'text': self.faiss_metadata[idx].get('text', ''),
                        'metadata': {k: v for k, v in self.faiss_metadata[idx].items() if k != 'text'},
                        'distance': float(distance)
                    }
                    retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the LLM and retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer
        """
        # Format context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            product = metadata.get('product', 'Unknown')
            context_parts.append(f"[Chunk {i} - Product: {product}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=query)
        
        # Generate answer
        if self.llm is not None:
            try:
                response = self.llm(
                    prompt,
                    max_length=len(prompt.split()) + 200,
                    num_return_sequences=1,
                    truncation=True
                )
                answer = response[0]['generated_text']
                # Extract only the answer part (after "Answer:")
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                return answer
            except Exception as e:
                print(f"Error generating answer with LLM: {e}")
                # Fallback to template-based response
                return self._generate_template_answer(query, retrieved_chunks)
        else:
            return self._generate_template_answer(query, retrieved_chunks)
    
    def _generate_template_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate a simple template-based answer when LLM is not available."""
        if not retrieved_chunks:
            return "I couldn't find any relevant complaints to answer your question."
        
        # Summarize the retrieved chunks
        products = [chunk.get('metadata', {}).get('product', 'Unknown') for chunk in retrieved_chunks]
        product_counts = pd.Series(products).value_counts()
        
        answer = f"Based on {len(retrieved_chunks)} relevant complaint(s), "
        answer += f"I found complaints primarily related to: {', '.join(product_counts.head(3).index.tolist())}. "
        answer += f"\n\nKey complaint excerpts:\n"
        
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            answer += f"\n{i}. {text}\n"
        
        return answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve and generate.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_chunks)
        
        # Format sources
        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            sources.append({
                'text': chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                'product': metadata.get('product', 'Unknown'),
                'issue': metadata.get('issue', 'Unknown'),
                'complaint_id': metadata.get('complaint_id', 'Unknown')
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }


def evaluate_rag_pipeline(
    rag: RAGPipeline,
    test_questions: List[str],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate the RAG pipeline on test questions.
    
    Args:
        rag: RAGPipeline instance
        test_questions: List of test questions
        output_file: Optional path to save evaluation results
        
    Returns:
        DataFrame with evaluation results
    """
    print("\n" + "="*60)
    print("RAG PIPELINE EVALUATION")
    print("="*60)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] Processing: {question}")
        
        result = rag.query(question)
        
        # Manual quality scoring (1-5)
        # In production, you'd use automated metrics
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Retrieved {result['num_sources']} sources")
        
        results.append({
            'Question': question,
            'Generated Answer': result['answer'],
            'Retrieved Sources': f"{result['num_sources']} sources",
            'Source 1 Preview': result['sources'][0]['text'][:150] + "..." if result['sources'] else "N/A",
            'Source 2 Preview': result['sources'][1]['text'][:150] + "..." if len(result['sources']) > 1 else "N/A",
            'Quality Score': None,  # To be filled manually
            'Comments': None  # To be filled manually
        })
    
    df_results = pd.DataFrame(results)
    
    if output_file:
        df_results.to_csv(output_file, index=False)
        print(f"\nEvaluation results saved to {output_file}")
    
    return df_results


def main():
    """Main function for testing the RAG pipeline."""
    # Initialize RAG pipeline
    # Try to load from pre-built embeddings first
    rag = RAGPipeline(
        vector_store_type="chromadb",  # Will auto-detect parquet if available
        vector_store_path="vector_store",
        top_k=5
    )
    
    # Test questions for evaluation
    test_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are the main issues with Personal Loans?",
        "What problems do customers face with Money Transfers?",
        "What are common complaints about Savings Accounts?",
        "What billing issues are customers experiencing?",
        "Are there any fraud-related complaints?",
        "What customer service problems are mentioned?",
        "Which product has the most complaints about fees?",
        "What are the top issues across all products?",
        "What do customers complain about regarding account access?"
    ]
    
    # Run evaluation
    results = evaluate_rag_pipeline(rag, test_questions, "data/processed/evaluation_results.csv")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nPlease review the results and add Quality Scores (1-5) and Comments manually.")


if __name__ == "__main__":
    main()

