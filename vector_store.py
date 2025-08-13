import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os

class VectorStore:
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "openai"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Choose embedding model
        if embedding_model == "openai" and os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings()
        else:
            # Use free sentence-transformers model
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Initialize or load existing vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one"""
        if os.path.exists(self.persist_directory):
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            except:
                # If loading fails, will create new one when documents added
                self.vectorstore = None
        else:
            # Will be created when documents are added
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            raise ValueError("No documents provided")
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        
        # Persist the changes
        self.vectorstore.persist()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever for the vector store"""
        if self.vectorstore is None:
            raise ValueError("No documents in vector store. Add documents first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None