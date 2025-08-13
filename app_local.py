import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store_completely_free import VectorStore
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot (Local/Free)",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

def simple_rag_response(vector_store, question: str, k: int = 3) -> str:
    """Simple RAG without OpenAI - just return relevant context"""
    try:
        # Get similar documents
        docs = vector_store.similarity_search(question, k=k)
        
        if not docs:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Create a simple response with context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"**Source {i}:**\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
        
        response = f"Based on your documents, here's what I found:\n\n"
        response += "\n\n---\n\n".join(context_parts)
        response += f"\n\n*Note: This is a simple context retrieval. For AI-generated answers, you need OpenAI credits.*"
        
        return response
        
    except Exception as e:
        return f"Error searching documents: {str(e)}"

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
        
        # Add documents to vector store (completely free)
        vector_store.add_documents(all_chunks)
        
        st.session_state.vector_store = vector_store
        st.session_state.documents_processed = True
        
        return len(all_chunks)
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return 0

def reset_system():
    """Reset the RAG system"""
    st.session_state.vector_store = None
    st.session_state.documents_processed = False
    st.session_state.messages = []

# Main app
def main():
    st.title("🤖 RAG Chatbot (Free/Local Version)")
    st.write("Upload documents and search through them - completely free!")
    
    st.info("ℹ️ This version uses free embeddings and simple context retrieval. No OpenAI API key required!")
    
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
        if st.session_state.documents_processed and st.session_state.vector_store:
            st.success("✅ Documents loaded")
            st.success("✅ Search system ready")
        else:
            st.warning("⚠️ No documents loaded")
            if not st.session_state.vector_store:
                st.info("💡 Upload documents to get started")
        
        # Chat controls
        if st.session_state.documents_processed:
            st.header("💬 Chat Controls")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Information
        st.header("ℹ️ How it works")
        st.write("• Documents are split into chunks")
        st.write("• Free embeddings index the content")
        st.write("• Search finds relevant passages")
        st.write("• No AI generation (free version)")
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.info("👈 Please upload documents using the sidebar to get started!")
        st.write("### How to use:")
        st.write("1. Upload PDF or TXT files using the sidebar")
        st.write("2. Click 'Process Documents' to index them")
        st.write("3. Ask questions to search through the content!")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Search your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                response = simple_rag_response(st.session_state.vector_store, prompt)
                st.markdown(response)
                
                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })

if __name__ == "__main__":
    main()