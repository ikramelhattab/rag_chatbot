import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os

class VectorStore:
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./chroma_db"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Use HuggingFace embeddings - completely free and offline after first download
        print("🔄 Initializing free embedding model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Free embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            # Fallback to basic embeddings
            from langchain.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
            print("⚠️ Using fallback embeddings")
        
        # Initialize or load existing vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                print("📂 Loaded existing vector database")
            except Exception as e:
                print(f"⚠️ Could not load existing database: {e}")
                self.vectorstore = None
        else:
            print("🏗️ Will create new vector database when documents are added")
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            raise ValueError("No documents provided")
        
        print(f"🔄 Processing {len(documents)} document chunks with free embeddings...")
        
        try:
            if self.vectorstore is None:
                print("🏗️ Creating new vector database...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
            else:
                print("📝 Adding documents to existing database...")
                self.vectorstore.add_documents(documents)
            
            # Persist the changes
            self.vectorstore.persist()
            print("✅ Documents successfully added to vector database")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise e
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            print("⚠️ No documents in vector store")
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"🔍 Found {len(results)} relevant documents")
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever for the vector store"""
        if self.vectorstore is None:
            raise ValueError("No documents in vector store. Add documents first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            if self.vectorstore is not None:
                self.vectorstore.delete_collection()
                self.vectorstore = None
            
            # Also remove the directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                
            print("🗑️ Vector database cleared")
        except Exception as e:
            print(f"⚠️ Error clearing database: {e}")