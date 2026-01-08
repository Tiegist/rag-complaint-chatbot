"""
Interactive Chat Interface for RAG Complaint Chatbot
Task 4: Build user-friendly interface using Gradio
"""

import gradio as gr
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline


# Initialize RAG pipeline (lazy loading)
rag_pipeline = None


def initialize_rag():
    """Initialize RAG pipeline on first use."""
    global rag_pipeline
    if rag_pipeline is None:
        print("Initializing RAG pipeline...")
        try:
            rag_pipeline = RAGPipeline(
                vector_store_type="chromadb",
                vector_store_path="vector_store",
                top_k=5,
                use_huggingface=True
            )
            print("RAG pipeline initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            return False
    return True


def chat_with_rag(message, history):
    """
    Process user message and generate response with RAG.
    
    Args:
        message: User's question
        history: Chat history (list of [user, bot] pairs)
        
    Returns:
        Updated history with bot response
    """
    # Initialize if needed
    if not initialize_rag():
        return history + [[message, "Error: Could not initialize RAG pipeline. Please check your vector store."]]
    
    if not message or message.strip() == "":
        return history
    
    try:
        # Query RAG pipeline
        result = rag_pipeline.query(message)
        
        # Format answer with sources
        answer = result['answer']
        
        # Add sources section
        sources_text = "\n\n**Sources:**\n"
        for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
            sources_text += f"\n**Source {i}** (Product: {source['product']}, Issue: {source['issue']}):\n"
            sources_text += f"> {source['text']}\n"
        
        full_response = answer + sources_text
        
        return history + [[message, full_response]]
    
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}\n\nPlease try rephrasing your question or check if the vector store is properly set up."
        return history + [[message, error_msg]]


def clear_chat():
    """Clear the chat history."""
    return []


def create_interface():
    """Create and launch the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(css=css, title="CrediTrust Complaint Analysis Chatbot") as demo:
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Complaint Analysis Chatbot
            
            Welcome! This AI-powered chatbot helps you analyze customer complaints across financial products.
            
            **How to use:**
            - Ask questions about customer complaints (e.g., "Why are people unhappy with Credit Cards?")
            - The chatbot will search through complaint data and provide evidence-based answers
            - Review the sources below each answer for verification
            
            **Supported Products:** Credit Cards, Personal Loans, Savings Accounts, Money Transfers
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about customer complaints...",
                        scale=4,
                        container=False
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Information")
                gr.Markdown(
                    """
                    **Example Questions:**
                    - Why are people unhappy with Credit Cards?
                    - What are the main issues with Personal Loans?
                    - What problems do customers face with Money Transfers?
                    - What are common complaints about Savings Accounts?
                    - Which product has the most billing issues?
                    """
                )
                
                gr.Markdown("### 📊 Status")
                status = gr.Textbox(
                    label="System Status",
                    value="Initializing...",
                    interactive=False
                )
        
        # Event handlers
        def update_status():
            if initialize_rag():
                return "✅ RAG Pipeline Ready"
            else:
                return "❌ RAG Pipeline Not Available"
        
        msg.submit(chat_with_rag, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        submit_btn.click(chat_with_rag, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        clear_btn.click(clear_chat, None, [chatbot])
        
        # Initialize and update status
        demo.load(update_status, None, [status])
    
    return demo


def main():
    """Main function to launch the interface."""
    print("="*60)
    print("Starting CrediTrust Complaint Analysis Chatbot")
    print("="*60)
    
    demo = create_interface()
    
    # Launch with sharing option (set share=False for local only)
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True
    )


if __name__ == "__main__":
    main()



