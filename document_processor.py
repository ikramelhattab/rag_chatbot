import os
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from various file types"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        documents = loader.load()
        return documents
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents.extend(txt_loader.load())
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_documents(self, source_path: str) -> List[Document]:
        """Complete document processing pipeline"""
        if os.path.isfile(source_path):
            documents = self.load_documents(source_path)
        elif os.path.isdir(source_path):
            documents = self.load_directory(source_path)
        else:
            raise ValueError(f"Invalid path: {source_path}")
        
        chunks = self.split_documents(documents)
        return chunks
