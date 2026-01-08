"""
Text chunking, embedding, and vector store indexing module.
Task 2: Convert cleaned text narratives into format suitable for semantic search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss
import pickle
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback implementation if langchain not available
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap, length_function, separators):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators
        
        def split_text(self, text):
            """Simple text splitter implementation."""
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - self.chunk_overlap
            return chunks
import json


class EmbeddingPipeline:
    """Handles text chunking, embedding generation, and vector store creation."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_type: str = "chromadb"  # or "faiss"
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            vector_store_type: Type of vector store ("chromadb" or "faiss")
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_type = vector_store_type
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Model loaded successfully!")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector store will be initialized later
        self.vector_store = None
        self.chroma_client = None
        self.chroma_collection = None
        
    def create_stratified_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 12000,
        product_col: str = "Product",
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a stratified sample ensuring proportional representation across products.
        
        Args:
            df: Input DataFrame with cleaned complaints
            sample_size: Target sample size (10K-15K)
            product_col: Name of the product column
            random_state: Random seed for reproducibility
            
        Returns:
            Stratified sample DataFrame
        """
        print(f"\nCreating stratified sample of {sample_size} complaints...")
        
        # Get product distribution
        product_counts = df[product_col].value_counts()
        print(f"\nOriginal distribution:")
        print(product_counts)
        
        # Calculate proportions
        total = len(df)
        proportions = product_counts / total
        
        # Calculate samples per product
        samples_per_product = {}
        remaining = sample_size
        
        for product in product_counts.index:
            count = product_counts[product]
            # Calculate proportional sample
            sample_count = max(1, int(proportions[product] * sample_size))
            # Don't exceed available records
            sample_count = min(sample_count, count)
            samples_per_product[product] = sample_count
            remaining -= sample_count
        
        # Distribute remaining samples to largest groups
        if remaining > 0:
            sorted_products = sorted(
                samples_per_product.items(),
                key=lambda x: product_counts[x[0]],
                reverse=True
            )
            for product, current_sample in sorted_products:
                if remaining <= 0:
                    break
                max_possible = product_counts[product]
                if current_sample < max_possible:
                    additional = min(remaining, max_possible - current_sample)
                    samples_per_product[product] += additional
                    remaining -= additional
        
        # Sample from each product
        sampled_dfs = []
        for product, n_samples in samples_per_product.items():
            product_df = df[df[product_col] == product]
            if len(product_df) >= n_samples:
                sampled = product_df.sample(n=n_samples, random_state=random_state)
                sampled_dfs.append(sampled)
                print(f"  {product}: {n_samples} samples")
        
        stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
        print(f"\nTotal stratified sample size: {len(stratified_sample)}")
        print(f"\nSample distribution:")
        print(stratified_sample[product_col].value_counts())
        
        return stratified_sample
    
    def chunk_text(self, text: str, complaint_id: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            complaint_id: ID of the complaint
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        chunk_objects = []
        for idx, chunk_text in enumerate(chunks):
            chunk_objects.append({
                'text': chunk_text,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'complaint_id': complaint_id
            })
        
        return chunk_objects
    
    def process_complaints(
        self,
        df: pd.DataFrame,
        narrative_col: str = "cleaned_narrative"
    ) -> List[Dict[str, Any]]:
        """
        Process all complaints: chunk and prepare for embedding.
        
        Args:
            df: DataFrame with complaints
            narrative_col: Name of the narrative column
            
        Returns:
            List of all chunks with metadata
        """
        print(f"\nProcessing {len(df)} complaints...")
        
        all_chunks = []
        
        # Find metadata columns
        metadata_cols = ['Product', 'Issue', 'Sub-issue', 'Company', 'State', 'Date received']
        available_metadata = {col: col for col in metadata_cols if col in df.columns}
        
        for idx, row in df.iterrows():
            complaint_id = str(row.get('Complaint ID', idx))
            narrative = str(row.get(narrative_col, ''))
            
            if not narrative or narrative == 'nan' or len(narrative.strip()) == 0:
                continue
            
            # Chunk the narrative
            chunks = self.chunk_text(narrative, complaint_id)
            
            # Add metadata to each chunk
            for chunk in chunks:
                # Add complaint metadata
                chunk['product_category'] = str(row.get('Product', 'Unknown'))
                chunk['product'] = str(row.get('Product', 'Unknown'))
                chunk['issue'] = str(row.get('Issue', 'Unknown'))
                chunk['sub_issue'] = str(row.get('Sub-issue', 'Unknown'))
                chunk['company'] = str(row.get('Company', 'Unknown'))
                chunk['state'] = str(row.get('State', 'Unknown'))
                chunk['date_received'] = str(row.get('Date received', 'Unknown'))
                
                all_chunks.append(chunk)
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def create_chromadb_store(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        persist_directory: str = "vector_store/chromadb"
    ):
        """
        Create and persist ChromaDB vector store.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            persist_directory: Directory to persist the database
        """
        print(f"\nCreating ChromaDB vector store...")
        
        # Create directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        collection_name = "complaint_chunks"
        try:
            self.chroma_collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
        
        # Prepare data for ChromaDB
        ids = [f"{chunk['complaint_id']}_{chunk['chunk_index']}" for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                'complaint_id': chunk['complaint_id'],
                'product_category': chunk['product_category'],
                'product': chunk['product'],
                'issue': chunk['issue'],
                'sub_issue': chunk['sub_issue'],
                'company': chunk['company'],
                'state': chunk['state'],
                'date_received': chunk['date_received'],
                'chunk_index': str(chunk['chunk_index']),
                'total_chunks': str(chunk['total_chunks'])
            }
            for chunk in chunks
        ]
        
        # Add to collection in batches (ChromaDB has batch size limits)
        print("Adding embeddings to ChromaDB in batches...")
        batch_size = 5000  # Safe batch size for ChromaDB
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            self.chroma_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            print(f"  Added batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({end_idx}/{total_chunks} chunks)")
        
        print(f"ChromaDB store created with {len(chunks)} chunks")
        print(f"Persisted to: {persist_directory}")
    
    def create_faiss_store(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        output_dir: str = "vector_store/faiss"
    ):
        """
        Create and persist FAISS vector store.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            output_dir: Directory to save FAISS index and metadata
        """
        print(f"\nCreating FAISS vector store...")
        
        # Create directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings
        index.add(embeddings.astype('float32'))
        
        # Save index
        index_path = Path(output_dir) / "faiss.index"
        faiss.write_index(index, str(index_path))
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = Path(output_dir) / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"Metadata saved to: {metadata_path}")
        
        # Save metadata as JSON for easier inspection
        json_path = Path(output_dir) / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)
        print(f"Metadata JSON saved to: {json_path}")
        
        print(f"FAISS store created with {len(chunks)} chunks")
    
    def run_pipeline(
        self,
        input_csv: str = "data/processed/filtered_complaints.csv",
        sample_size: int = 12000,
        output_dir: str = "vector_store"
    ):
        """
        Run the complete pipeline: load, sample, chunk, embed, and index.
        
        Args:
            input_csv: Path to cleaned complaints CSV
            sample_size: Size of stratified sample
            output_dir: Directory for vector store
        """
        print("="*60)
        print("EMBEDDING PIPELINE")
        print("="*60)
        
        # Load data
        print(f"\nLoading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} complaints")
        
        # Create stratified sample
        sampled_df = self.create_stratified_sample(df, sample_size=sample_size)
        
        # Process complaints into chunks
        chunks = self.process_complaints(sampled_df)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Create vector store
        if self.vector_store_type == "chromadb":
            self.create_chromadb_store(chunks, embeddings, f"{output_dir}/chromadb")
        else:
            self.create_faiss_store(chunks, embeddings, f"{output_dir}/faiss")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)


def main():
    """Main function to run the embedding pipeline."""
    pipeline = EmbeddingPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        vector_store_type="chromadb"  # Change to "faiss" if preferred
    )
    
    pipeline.run_pipeline(
        input_csv="data/processed/filtered_complaints.csv",
        sample_size=12000
    )


if __name__ == "__main__":
    main()

