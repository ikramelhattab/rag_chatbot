from langchain.llms.openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

class RAGChain:
    def __init__(self, 
                 vector_store: VectorStore,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1):
        
        self.vector_store = vector_store
        
        # Initialize LLM
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on the provided context. 
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Helpful Answer:""",
            input_variables=["context", "question"]
        )
        
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the RAG chain"""
        try:
            retriever = self.vector_store.get_retriever()
            
            # Create the RAG chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
        except Exception as e:
            raise ValueError(f"Failed to setup RAG chain: {str(e)}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def chat(self, question: str) -> str:
        """Simple chat interface"""
        result = self.query(question)
        return result["answer"]
    
    def get_sources(self, question: str) -> List[str]:
        """Get source documents for a question"""
        result = self.query(question)
        if result["success"]:
            return [doc.page_content for doc in result["source_documents"]]
        return []