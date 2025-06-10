#!/usr/bin/env python3
"""
Setup script for LLM Fact Checker
Automates initial setup including model downloads and dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def setup_environment():
    """Set up the development environment"""
    print("=" * 60)
    print("🔍 LLM FACT CHECKER SETUP")
    print("=" * 60)
    print("Setting up your fact-checking environment...")
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create necessary directories
    print("📁 Creating directories...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("✅ Directories created")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("💡 Tip: Try using 'pip install --user -r requirements.txt' if permission denied")
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️ spaCy model download failed - the system will still work with limited NLP capabilities")
    
    # Test imports
    print("🧪 Testing imports...")
    try:
        import streamlit
        import sentence_transformers
        import faiss
        import spacy
        import feedparser
        import transformers
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating environment configuration...")
        with open(env_file, "w") as f:
            f.write("# LLM Fact Checker Configuration\n")
            f.write("# You can add API keys and other settings here\n")
            f.write("\n")
            f.write("# Example:\n")
            f.write("# OPENAI_API_KEY=your_api_key_here\n")
            f.write("# HUGGINGFACE_API_KEY=your_api_key_here\n")
        print("✅ Created .env file")
    
    return True

def test_system():
    """Test if the system is working"""
    print("\n🧪 Testing system functionality...")
    
    try:
        from src.main_pipeline import FactCheckingPipeline
        
        # Quick test
        print("📦 Initializing pipeline...")
        pipeline = FactCheckingPipeline()
        
        # Test claim extraction
        print("🔍 Testing claim extraction...")
        claim_extractor = pipeline.claim_extractor
        test_claim = claim_extractor.get_primary_claim("The government announced a new policy.")
        
        if test_claim:
            print("✅ Claim extraction working")
        else:
            print("⚠️ Claim extraction test failed")
        
        print("✅ Basic system test passed")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 Next steps:")
    print()
    print("1. Run the demo to test the system:")
    print("   python demo.py")
    print()
    print("2. Start the web interface:")
    print("   streamlit run app.py")
    print()
    print("3. Or use the Python API:")
    print("   from src.main_pipeline import FactCheckingPipeline")
    print("   pipeline = FactCheckingPipeline()")
    print("   result = pipeline.fact_check_text('Your claim here')")
    print()
    print("📚 For more information, see README.md")
    print()
    print("🆘 If you encounter issues:")
    print("   - Check the troubleshooting section in README.md")
    print("   - Ensure you have a stable internet connection")
    print("   - Try running: pip install --upgrade -r requirements.txt")
    print()

def main():
    """Main setup function"""
    try:
        # Setup environment
        if not setup_environment():
            print("\n❌ Setup failed. Please check the errors above.")
            return False
        
        # Test system
        if not test_system():
            print("\n⚠️ Setup completed but system test failed.")
            print("The system may still work, but please check for issues.")
        
        # Print next steps
        print_next_steps()
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 