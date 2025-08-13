#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running {command}: {e}")
        return False

def fix_dependencies():
    """Fix dependency issues"""
    print("🔧 Fixing dependency compatibility issues...")
    
    # Uninstall problematic packages
    print("📦 Uninstalling incompatible packages...")
    uninstall_commands = [
        "pip uninstall -y openai langchain langchain-openai",
    ]
    
    for cmd in uninstall_commands:
        run_command(cmd)
    
    # Install compatible versions
    print("📦 Installing compatible versions...")
    install_commands = [
        "pip install openai==0.28.1",
        "pip install langchain==0.0.352",
        "pip install langchain-openai==0.0.2",
        "pip install --upgrade chromadb==0.4.22",
        "pip install --upgrade sentence-transformers==2.2.2",
        "pip install --upgrade streamlit==1.29.0",
        "pip install --upgrade pypdf2==3.0.1",
        "pip install --upgrade python-dotenv==1.0.0",
        "pip install --upgrade tiktoken==0.5.2"
    ]
    
    success = True
    for cmd in install_commands:
        if not run_command(cmd):
            success = False
    
    if success:
        print("✅ Dependencies fixed successfully!")
        print("Now you can run: streamlit run app.py")
    else:
        print("❌ Some issues occurred. Try manual installation.")
        print("Manual commands:")
        for cmd in install_commands:
            print(f"  {cmd}")

if __name__ == "__main__":
    fix_dependencies()