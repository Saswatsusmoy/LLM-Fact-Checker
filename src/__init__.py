# LLM Fact Checker Package

__version__ = "1.0.0"
__author__ = "LLM Fact Checker Team"
__description__ = "AI-Powered Fact Checking System using RAG and Local LLMs"

from .claim_extractor import ClaimExtractor
from .fact_database import FactDatabase
from .vector_retrieval import VectorRetrieval
from .fact_checker import FactChecker
from .main_pipeline import FactCheckingPipeline

__all__ = [
    "ClaimExtractor",
    "FactDatabase", 
    "VectorRetrieval",
    "FactChecker",
    "FactCheckingPipeline"
] 