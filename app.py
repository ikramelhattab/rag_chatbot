import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

def check_api_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        return False
    return True

def process_documents(uploaded_files):
    """Process uploaded documents"""
    try:
        processor = DocumentProcessor()
        vector_store = VectorStore()
        
        all_chunks = []
        
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process document
                chunks = processor.process_documents(temp_file_path)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            st.error("No content could be extracted from the uploaded files.")
            return 0
        
        # Add documents to vector store
        vector_store.add_documents(all_chunks)
        
        # Initialize RAG chain
        st.session_state.rag_chain = RAGChain(vector_store)
        st.session_state.documents_processed = True
        
        return len(all_chunks)
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return 0

def reset_system():
    """Reset the RAG system"""
    st.session_state.rag_chain = None
    st.session_state.documents_processed = False
    st.session_state.messages = []
    
    # Clean up vector store
    try:
        vector_store = VectorStore()
        vector_store.delete_collection()
    except:
        pass

# Main app
def main():
    st.title("🤖 RAG Chatbot")
    st.write("Upload documents and ask questions based on their content!")
    
    # Check API key
    if not check_api_key():
        st.error("⚠️ OpenAI API key not found!")
        st.write("Please:")
        st.write("1. Copy `.env.template` to `.env`")
        st.write("2. Add your OpenAI API key to the `.env` file")
        st.write("3. Restart the application")
        return
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Reset button
        if st.button("🔄 Reset System", help="Clear all documents and chat history"):
            reset_system()
            st.rerun()
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to create your knowledge base"
        )
        
        if uploaded_files and st.button("📚 Process Documents"):
            with st.spinner("Processing documents..."):
                num_chunks = process_documents(uploaded_files)
                if num_chunks > 0:
                    st.success(f"✅ Processed {len(uploaded_files)} files into {num_chunks} chunks!")
                else:
                    st.error("❌ Failed to process documents")
        
        # System status
        st.header("📊 System Status")
        if st.session_state.documents_processed and st.session_state.rag_chain:
            st.success("✅ Documents loaded")
            st.success("✅ RAG system ready")
        else:
            st.warning("⚠️ No documents loaded")
            if not st.session_state.rag_chain:
                st.info("💡 Upload documents to get started")
        
        # Chat controls
        if st.session_state.documents_processed:
            st.header("💬 Chat Controls")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.info("👈 Please upload documents using the sidebar to get started!")
        st.write("### How to use:")
        st.write("1. Upload PDF or TXT files using the sidebar")
        st.write("2. Click 'Process Documents' to index them")
        st.write("3. Ask questions about the content!")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(source.page_content[:500] + ("..." if len(source.page_content) > 500 else ""))
                        if hasattr(source, 'metadata') and source.metadata:
                            st.caption(f"From: {source.metadata.get('source', 'Unknown')}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_chain.query(prompt)
                
                if result["success"]:
                    st.markdown(result["answer"])
                    
                    # Store message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["source_documents"]
                    })
                    
                    # Show sources
                    if result["source_documents"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(result["source_documents"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(source.page_content[:500] + ("..." if len(source.page_content) > 500 else ""))
                                if hasattr(source, 'metadata') and source.metadata:
                                    st.caption(f"From: {source.metadata.get('source', 'Unknown')}")
                                st.divider()
                else:
                    st.error(result["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": []
                    })

if __name__ == "__main__":
    main()
