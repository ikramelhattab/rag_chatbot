# RAG Chatbot 🤖

A complete Retrieval Augmented Generation (RAG) chatbot implementation that allows you to chat with your documents using OpenAI's language models.

## Features ✨

- **Document Support**: PDF and TXT files
- **Smart Chunking**: Intelligent document splitting for optimal retrieval
- **Vector Search**: ChromaDB for efficient similarity search
- **Multiple Interfaces**: Both web (Streamlit) and command-line interfaces
- **Source Attribution**: Shows which documents informed each answer
- **Persistent Storage**: Documents stay indexed between sessions

## Quick Start 🚀

### 1. Setup
```bash
# Clone or download all files to a folder
# Run the setup script
python setup.py
```

### 2. Configure API Key
Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Add Documents
Place your PDF or TXT files in the `documents/` folder.

### 4. Run the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python cli_chat.py documents/
```

## File Structure 📁

```
rag-chatbot/
├── app.py                 # Streamlit web interface
├── cli_chat.py           # Command line interface
├── document_processor.py # Document loading and chunking
├── vector_store.py       # Vector database management
├── rag_chain.py          # RAG logic and LLM integration
├── setup.py              # Automated setup script
├── requirements.txt      # Python dependencies
├── .env.template         # Environment template
├── .env                  # Your API keys (create from template)
├── README.md            # This file
├── documents/           # Put your documents here
└── chroma_db/           # Vector database storage
```

## Usage Guide 📖

### Web Interface
1. Run `streamlit run app.py`
2. Upload documents using
