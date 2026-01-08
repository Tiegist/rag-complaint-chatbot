"""
Helper script to run the complete RAG pipeline from start to finish.
This script orchestrates all tasks in sequence.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def run_task1():
    """Run Task 1: EDA and Preprocessing."""
    print("\n" + "="*60)
    print("TASK 1: EDA AND PREPROCESSING")
    print("="*60)
    
    from data_processing import ComplaintDataProcessor
    
    data_path = 'data/raw/complaints.csv'
    processor = ComplaintDataProcessor(data_path)
    
    try:
        processor.load_data()
        processor.perform_eda()
        processor.filter_data()
        processor.preprocess_data()
        processor.save_cleaned_data()
        print("\n✅ Task 1 completed successfully!")
        return True
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find data file at {data_path}")
        print("Please download the CFPB complaint dataset and place it in data/raw/")
        return False
    except Exception as e:
        print(f"\n❌ Error in Task 1: {e}")
        return False


def run_task2():
    """Run Task 2: Embedding Pipeline."""
    print("\n" + "="*60)
    print("TASK 2: EMBEDDING PIPELINE")
    print("="*60)
    
    from embedding_pipeline import EmbeddingPipeline
    
    try:
        pipeline = EmbeddingPipeline(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50,
            vector_store_type="chromadb"
        )
        
        pipeline.run_pipeline(
            input_csv="data/processed/filtered_complaints.csv",
            sample_size=12000
        )
        print("\n✅ Task 2 completed successfully!")
        return True
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run Task 1 first to generate filtered_complaints.csv")
        return False
    except Exception as e:
        print(f"\n❌ Error in Task 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_task3():
    """Run Task 3: RAG Pipeline Evaluation."""
    print("\n" + "="*60)
    print("TASK 3: RAG PIPELINE EVALUATION")
    print("="*60)
    
    from rag_pipeline import RAGPipeline, evaluate_rag_pipeline
    
    try:
        rag = RAGPipeline(
            vector_store_type="chromadb",
            vector_store_path="vector_store",
            top_k=5
        )
        
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
        
        results = evaluate_rag_pipeline(
            rag,
            test_questions,
            "data/processed/evaluation_results.csv"
        )
        
        print("\n✅ Task 3 completed successfully!")
        print(f"Evaluation results saved to data/processed/evaluation_results.csv")
        return True
    except Exception as e:
        print(f"\n❌ Error in Task 3: {e}")
        print("Make sure you have:")
        print("  1. Run Task 2 to create vector store, OR")
        print("  2. Placed complaint_embeddings.parquet in vector_store/")
        import traceback
        traceback.print_exc()
        return False


def run_task4():
    """Run Task 4: Launch Chat Interface."""
    print("\n" + "="*60)
    print("TASK 4: LAUNCHING CHAT INTERFACE")
    print("="*60)
    
    try:
        import app
        print("\n✅ Starting Gradio interface...")
        print("Open your browser to http://localhost:7860")
        app.main()
    except Exception as e:
        print(f"\n❌ Error in Task 4: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run tasks."""
    parser = argparse.ArgumentParser(description="Run RAG Complaint Chatbot Pipeline")
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific task (1-4). If not specified, runs all tasks."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks in sequence"
    )
    
    args = parser.parse_args()
    
    if args.task:
        tasks = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}
        tasks[args.task]()
    elif args.all:
        print("Running all tasks in sequence...")
        success = True
        success = success and run_task1()
        if success:
            success = success and run_task2()
        if success:
            success = success and run_task3()
        if success:
            print("\n" + "="*60)
            print("All tasks completed! You can now run Task 4 to launch the UI:")
            print("  python run_pipeline.py --task 4")
            print("="*60)
    else:
        print("RAG Complaint Chatbot Pipeline")
        print("\nUsage:")
        print("  python run_pipeline.py --task 1  # Run Task 1 only")
        print("  python run_pipeline.py --task 2  # Run Task 2 only")
        print("  python run_pipeline.py --task 3  # Run Task 3 only")
        print("  python run_pipeline.py --task 4  # Run Task 4 (launch UI)")
        print("  python run_pipeline.py --all     # Run all tasks (1-3)")


if __name__ == "__main__":
    main()



