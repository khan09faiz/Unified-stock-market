"""
Setup script for the Unified Stock Market Analysis System.
Run this script to set up the environment and install dependencies.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    """Set up the environment for the stock analysis system."""
    print("🚀 Setting up Unified Stock Market Analysis System")
    print("=" * 50)
    
    # Create directories
    directories = ["results", "data", "models"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
    
    # Install core dependencies
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.87",
        "scikit-learn>=1.0.0"
    ]
    
    print("\\n📦 Installing core packages...")
    for package in core_packages:
        try:
            install_package(package)
            print(f"✅ Installed: {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
    
    # Optional packages (may fail without errors)
    optional_packages = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "nltk>=3.7",
        "statsmodels>=0.13.0",
        "arch>=5.3.0",
        "gym>=0.21.0",
        "optuna>=3.0.0",
        "tqdm>=4.62.0",
        "wordcloud>=1.8.2"
    ]
    
    print("\\n📦 Installing optional packages...")
    for package in optional_packages:
        try:
            install_package(package)
            print(f"✅ Installed: {package}")
        except Exception as e:
            print(f"⚠️  Optional package {package} failed: {e}")
    
    # Download NLTK data
    print("\\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'], quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    print("\\n🎉 Setup completed!")
    print("\\nTo get started, run:")
    print("  python demo.py          # Quick demo")
    print("  python main.py          # Full analysis")

if __name__ == "__main__":
    setup_environment()
