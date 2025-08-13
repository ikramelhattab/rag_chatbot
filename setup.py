#!/usr/bin/env python3

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def setup_env_file():
    """Setup environment file"""
    env_file = ".env"
    template_file = ".env.template"
    
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists(template_file):
        print("📝 Creating .env file from template...")
        try:
            with open(template_file, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ .env file created")
            print("⚠️  Please edit .env and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ .env.template file not found")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["documents", "chroma_db"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def main():
    print("🚀 RAG Chatbot Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed. Please install packages manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup environment file
    if not setup_env_file():
        print("\n❌ Failed to setup environment file")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Put your documents in the 'documents' folder")
    print("3. Run the application:")
    print("   - Web interface: streamlit run app.py")
    print("   - Command line: python cli_chat.py documents/")

if __name__ == "__main__":
    main()
