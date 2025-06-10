from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-fact-checker",
    version="1.0.0",
    author="LLM Fact Checker Team",
    author_email="your.email@example.com",
    description="AI-Powered Fact Checking System using RAG and Local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-fact-checker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "faiss-gpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fact-checker=src.main_pipeline:main",
        ],
    },
    keywords="fact-checking, nlp, rag, llm, ai, verification, claims",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-fact-checker/issues",
        "Source": "https://github.com/yourusername/llm-fact-checker",
        "Documentation": "https://github.com/yourusername/llm-fact-checker#readme",
    },
    include_package_data=True,
    zip_safe=False,
) 