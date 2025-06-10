#!/usr/bin/env python3
"""
Demo script for LLM Fact Checker
Showcases the fact-checking system with example claims
"""

import os
import sys
import time
from src.main_pipeline import FactCheckingPipeline

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🔍 LLM FACT CHECKER DEMO")
    print("=" * 60)
    print("AI-Powered Claim Verification System")
    print("Using RAG + Local LLMs")
    print("=" * 60)
    print()

def print_result(claim, result):
    """Print formatted fact-check result"""
    print(f"📝 CLAIM: {claim}")
    print("-" * 50)
    print(f"🎯 VERDICT: {result['overall_verdict']}")
    print(f"📊 CONFIDENCE: {result.get('overall_confidence', 0):.2f}")
    print(f"💭 REASONING: {result['reasoning']}")
    
    if result.get('results') and result['results']:
        evidence = result['results'][0].get('evidence', [])
        if evidence:
            print(f"📋 EVIDENCE ({len(evidence)} sources):")
            for i, ev in enumerate(evidence[:2], 1):
                print(f"   {i}. {ev[:100]}...")
    
    print("=" * 60)
    print()

def run_demo():
    """Run the fact-checking demo"""
    print_banner()
    
    # Test claims - mix of different types
    test_claims = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "India's GDP grew by 7.6% in the second quarter of fiscal 2023-24.",
        "Climate change is causing increased rainfall patterns across India.",
        "The government launched a new digital currency called e-Rupee in 2023.",
        "Over 130 crore Aadhaar cards have been issued in India.",
        "Scientists have discovered a cure for all types of cancer.",
        "The Earth is flat and NASA has been lying to us."
    ]
    
    print("🚀 Initializing LLM Fact Checker...")
    print("This may take a few minutes on first run (downloading models)...")
    print()
    
    try:
        # Initialize pipeline
        pipeline = FactCheckingPipeline()
        
        print("📚 Setting up fact database...")
        print("Fetching latest facts from trusted news sources...")
        pipeline.setup_database(update_facts=True, max_facts=30)
        
        # Get database stats
        stats = pipeline.get_database_stats()
        print(f"✅ Database ready: {stats['fact_database']['total_facts']} facts loaded")
        print(f"🗂️ Vector index: {stats['vector_index']['total_facts']} embeddings")
        print()
        
        print("🔍 Starting fact-checking demo...")
        print("Testing various types of claims...")
        print()
        
        # Process each test claim
        results = []
        for i, claim in enumerate(test_claims, 1):
            print(f"[{i}/{len(test_claims)}] Processing claim...")
            
            start_time = time.time()
            result = pipeline.fact_check_text(claim)
            end_time = time.time()
            
            print_result(claim, result)
            results.append(result)
            
            # Show processing time
            print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
            print()
            
            # Brief pause between claims
            time.sleep(1)
        
        # Show summary statistics
        print("📊 DEMO SUMMARY")
        print("-" * 30)
        
        verdicts = [r.get('overall_verdict', '🤷‍♂️ Unverifiable') for r in results]
        true_count = sum(1 for v in verdicts if '✅' in v)
        false_count = sum(1 for v in verdicts if '❌' in v)
        unverifiable_count = sum(1 for v in verdicts if '🤷‍♂️' in v)
        
        print(f"Total claims tested: {len(results)}")
        print(f"✅ True: {true_count}")
        print(f"❌ False: {false_count}")
        print(f"🤷‍♂️ Unverifiable: {unverifiable_count}")
        
        avg_confidence = sum(r.get('overall_confidence', 0) for r in results) / len(results)
        print(f"📈 Average confidence: {avg_confidence:.2f}")
        
        print()
        print("🎉 Demo completed successfully!")
        print()
        print("🚀 To start the web interface, run:")
        print("   streamlit run app.py")
        print()
        print("📚 For more information, see README.md")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Download spaCy model: python -m spacy download en_core_web_sm")
        print("3. Check internet connection for RSS feed access")
        print("4. Try running with fewer claims or smaller models")
        return False
    
    return True

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run demo
    success = run_demo()
    sys.exit(0 if success else 1) 