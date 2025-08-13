#!/usr/bin/env python3

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import os
from dotenv import load_dotenv
import sys

def check_api_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ Error: OpenAI API key not found!")
        print("\nPlease:")
        print("1. Copy .env.template to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Run this script again")
        return False
    return True

def setup_rag_system(document_path: str):
    """Setup the RAG system with documents"""
    print("🔧 Setting up RAG system...")
    
    try:
        # Process documents
        processor = DocumentProcessor()
        chunks = processor.process_documents(document_path)
        print(f"📄 Processed {len(chunks)} document chunks")
        
        # Create vector store
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        print("💾 Documents added to vector store")
        
        # Create RAG chain
        rag_chain = RAGChain(vector_store)
        print("✅ RAG system ready!")
        
        return rag_chain
        
    except Exception as e:
        print(f"❌ Error setting up RAG system: {str(e)}")
        return None

def chat_loop(rag_chain: RAGChain):
    """Main chat loop"""
    print("\n🤖 RAG Chatbot ready!")
    print("Type your questions below. Commands:")
    print("  'quit', 'exit', 'bye' - Exit the chatbot")
    print("  'sources' - Show sources for last answer")
    print("  'help' - Show this help message")
    print("-" * 50)
    
    last_result = None
    
    while True:
        try:
            question = input("\n👤 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n💡 Available commands:")
                print("  - Type any question about your documents")
                print("  - 'sources' - Show sources for the last answer")
                print("  - 'quit' or 'exit' - Exit the chatbot")
                continue
            
            if question.lower() == 'sources':
                if last_result and last_result.get("source_documents"):
                    print("\n📚 Sources from last answer:")
                    for i, source in enumerate(last_result["source_documents"], 1):
                        print(f"\n--- Source {i} ---")
                        print(source.page_content[:300] + ("..." if len(source.page_content) > 300 else ""))
                        if hasattr(source, 'metadata') and source.metadata:
                            print(f"From: {source.metadata.get('source', 'Unknown')}")
                else:
                    print("❌ No sources available from the last answer.")
                continue
            
            if not question:
                continue
            
            print("🤖 Bot: ", end="", flush=True)
            result = rag_chain.query(question)
            last_result = result
            
            if result["success"]:
                print(result["answer"])
                
                # Show number of sources
                if result["source_documents"]:
                    print(f"\n📖 (Answer based on {len(result['source_documents'])} sources - type 'sources' to view)")
            else:
                print(f"❌ {result['answer']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    print("🚀 RAG Chatbot CLI")
    print("=" * 30)
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Get document path from user
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
        document_path = input("📁 Enter path to your documents (file or directory): ").strip()
    
    if not os.path.exists(document_path):
        print(f"❌ Error: Path '{document_path}' does not exist")
        sys.exit(1)
    
    try:
        # Setup RAG system
        rag_chain = setup_rag_system(document_path)
        
        if rag_chain is None:
            print("❌ Failed to setup RAG system. Exiting.")
            sys.exit(1)
        
        # Start chat
        chat_loop(rag_chain)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
