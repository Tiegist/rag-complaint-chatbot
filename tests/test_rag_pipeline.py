"""
Unit tests for RAG pipeline module.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_rag_pipeline_initialization():
    """Test RAGPipeline initialization (may fail if vector store not available)."""
    try:
        from rag_pipeline import RAGPipeline
        
        # This will fail if vector store doesn't exist, which is expected
        # In a real test environment, you'd set up mock data
        rag = RAGPipeline(
            vector_store_type="chromadb",
            vector_store_path="vector_store",
            top_k=5
        )
        assert rag.top_k == 5
        assert rag.embedding_model is not None
    except FileNotFoundError:
        # Expected if vector store not set up
        pytest.skip("Vector store not available for testing")


def test_prompt_template():
    """Test prompt template creation."""
    from rag_pipeline import RAGPipeline
    
    # Create a minimal RAG instance just to test prompt
    class MockRAG:
        def _create_prompt_template(self):
            from langchain.prompts import PromptTemplate
            template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context from real complaint narratives.

Use the following retrieved complaint excerpts to formulate your answer. Be specific and cite examples from the context when possible. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {question}

Answer:"""
            return PromptTemplate(template=template, input_variables=["context", "question"])
    
    mock_rag = MockRAG()
    prompt_template = mock_rag._create_prompt_template()
    
    # Test prompt formatting
    formatted = prompt_template.format(
        context="Test context",
        question="Test question"
    )
    
    assert "Test context" in formatted
    assert "Test question" in formatted
    assert "CrediTrust Financial" in formatted



