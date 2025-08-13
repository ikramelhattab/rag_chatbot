#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command, ignore_errors=False):
    """Run a command and return success status"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_errors:
            print(f"Error: {result.stderr}")
            return False
        print(f"Success: {result.stdout}")
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

def install_free_dependencies():
    """Install only free dependencies"""
    print("🔧 Installing completely free RAG chatbot dependencies...")
    
    # Remove any OpenAI related packages
    print("🗑️ Removing OpenAI dependencies...")
    uninstall_commands = [
        "pip uninstall -y openai langchain-openai",
    ]
    
    for cmd in uninstall_commands:
        run_command(cmd, ignore_errors=True)  # Ignore if not installed
    
    # Install free packages only
    print("📦 Installing free packages...")
    install_commands = [
        "pip install langchain==0.0.352",
        "pip install chromadb==0.4.22", 
        "pip install sentence-transformers==2.2.2",
        "pip install streamlit==1.29.0",
        "pip install pypdf2==3.0.1",
        "pip install python-dotenv==1.0.0",
        "pip install tiktoken==0.5.2",
        "pip install transformers==4.36.2",
        "pip install torch==2.1.2",
        "pip install huggingface-hub==0.19.4"
    ]
    
    success = True
    for cmd in install_commands:
        if not run_command(cmd):
            success = False
    
    if success:
        print("✅ Free dependencies installed successfully!")
        print("\n🚀 You can now run:")
        print("   streamlit run app_local.py")
        print("\n💡 This version is completely free - no OpenAI API needed!")
    else:
        print("❌ Some installations failed. Try running commands manually.")

def create_sample_documents():
    """Create sample documents for testing"""
    if not os.path.exists("documents"):
        os.makedirs("documents")
        
    sample_content = """
# Sample Document for RAG Testing

This is a sample document to test your RAG chatbot.

## About RAG
Retrieval Augmented Generation (RAG) is a technique that combines:
- Information retrieval from a knowledge base
- Natural language generation using AI models

## Benefits of RAG
1. Provides accurate, source-based answers
2. Reduces hallucination in AI responses
3. Allows chatbots to work with specific documents
4. Can work with constantly updated information

## How it works
1. Documents are split into chunks
2. Chunks are converted to embeddings
3. User questions are matched to relevant chunks
4. AI generates responses based on retrieved context

Feel free to ask questions about this document!
"""
    
    with open("documents/sample_rag_info.txt", "w") as f:
        f.write(sample_content)
    
    print("📄 Created sample document: documents/sample_rag_info.txt")

if __name__ == "__main__":
    print("🤖 RAG Chatbot Free Version Setup")
    print("=" * 40)
    
    install_free_dependencies()
    create_sample_documents()
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Run: streamlit run app_local.py")
    print("2. Upload your own documents or use the sample")
    print("3. Ask questions about your documents!")
    print("\n💰 Cost: $0 - completely free!")