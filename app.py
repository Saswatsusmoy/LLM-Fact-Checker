import streamlit as st
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List

# Fix for PyTorch Streamlit compatibility issue
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Set environment variable to prevent torch classes error
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

# Prevent Streamlit from watching torch modules
try:
    import torch
    torch.jit._state._enabled = False
except:
    pass

# Import our fact-checking pipeline
from src.main_pipeline import FactCheckingPipeline

# Page configuration
st.set_page_config(
    page_title="LLM Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .fact-check-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .verdict-true {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .verdict-false {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .verdict-unverifiable {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: #e9ecef;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load the fact-checking pipeline (cached for performance)"""
    return FactCheckingPipeline()

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'fact_check_history' not in st.session_state:
        st.session_state.fact_check_history = []
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []

def setup_database_ui():
    """UI for database setup and configuration"""
    st.sidebar.header("🗄️ Enhanced Database Setup")
    
    with st.sidebar.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Enhanced database update controls
        st.markdown("**📡 RSS Feed Retrieval**")
        update_db = st.button("🔄 Update Fact Database", key="update_db")
        
        # Advanced options
        col1, col2 = st.columns(2)
        with col1:
            max_facts = st.slider("Target Facts", 100, 1000, 500, 50)
        with col2:
            update_mode = st.selectbox("Update Mode", ["Standard", "Quick (200)", "Deep (800)"])
        
        # Map update mode to actual target
        mode_mapping = {"Standard": max_facts, "Quick (200)": 200, "Deep (800)": 800}
        actual_target = mode_mapping[update_mode]
        
        if update_db:
            with st.spinner(f"Updating database with {actual_target} target facts..."):
                try:
                    st.session_state.pipeline.setup_database(
                        update_facts=True, 
                        max_facts=actual_target
                    )
                    st.success(f"✅ Database updated successfully!")
                    st.info(f"🎯 Target: {actual_target} facts")
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")
        
        # Enhanced database statistics
        if st.session_state.pipeline:
            try:
                stats = st.session_state.pipeline.get_database_stats()
                fact_stats = stats['fact_database']
                vector_stats = stats['vector_retrieval']
                
                # Main metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📰 Total Facts", fact_stats['total_facts'])
                    st.metric("🔍 Vector Index", vector_stats['total_facts'])
                
                with col2:
                    st.metric("📅 Recent Facts", fact_stats.get('recent_facts', 0))
                    st.metric("📊 Avg Length", f"{fact_stats.get('avg_content_length', 0)} chars")
                
                # Source breakdown
                if fact_stats.get('sources'):
                    st.markdown("**📰 News Sources:**")
                    sources = fact_stats['sources']
                    top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
                    for source, count in top_sources:
                        st.text(f"  • {source}: {count}")
                    
                    if len(sources) > 5:
                        st.text(f"  ... and {len(sources) - 5} more sources")
                
                # RSS feed status indicator
                if fact_stats['total_facts'] >= 200:
                    st.success("🟢 RSS Feeds: Excellent")
                elif fact_stats['total_facts'] >= 100:
                    st.warning("🟡 RSS Feeds: Good")
                else:
                    st.error("🔴 RSS Feeds: Limited")
                    
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def setup_configuration_ui():
    """UI for pipeline configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        if st.session_state.pipeline:
            config = st.session_state.pipeline.config
            
            # Configuration sliders
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                0.1, 1.0, 
                config['similarity_threshold'], 
                0.1
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.1, 1.0, 
                config['confidence_threshold'], 
                0.1
            )
            
            max_evidence = st.slider(
                "Max Evidence Facts", 
                1, 10, 
                config['max_evidence_facts']
            )
            
            # Update configuration
            if st.button("Apply Configuration"):
                new_config = {
                    'similarity_threshold': similarity_threshold,
                    'confidence_threshold': confidence_threshold,
                    'max_evidence_facts': max_evidence
                }
                st.session_state.pipeline.update_configuration(new_config)
                st.success("Configuration updated!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_verdict_card(result: Dict):
    """Render a fact-check result as a styled card"""
    verdict = result.get('overall_verdict', '🤷‍♂️ Unverifiable')
    confidence = result.get('overall_confidence', 0.0)
    reasoning = result.get('reasoning', 'No reasoning provided')
    
    # Determine card style based on verdict
    if '✅' in verdict:
        card_class = "fact-check-card verdict-true"
    elif '❌' in verdict:
        card_class = "fact-check-card verdict-false"
    else:
        card_class = "fact-check-card verdict-unverifiable"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {verdict}")
        st.markdown(f"**Reasoning:** {reasoning}")
    
    with col2:
        # Confidence meter
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Evidence section
    if result.get('results') and result['results'][0].get('evidence'):
        with st.expander("📋 View Evidence Details"):
            evidence_list = result['results'][0]['evidence']
            sources = result['results'][0].get('evidence_sources', [])
            scores = result['results'][0].get('similarity_scores', [])
            
            for i, (evidence, source, score) in enumerate(zip(evidence_list[:3], sources, scores)):
                st.markdown(f"**Source {i+1}:** {source} (Similarity: {score:.2f})")
                st.markdown(f"> {evidence[:200]}...")
                st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_feedback_section(result_index: int):
    """Render feedback section for a fact-check result"""
    st.markdown("### 💭 Was this helpful?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Helpful", key=f"helpful_{result_index}"):
            feedback = {
                'result_index': result_index,
                'feedback': 'helpful',
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.feedback_data.append(feedback)
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{result_index}"):
            feedback = {
                'result_index': result_index,
                'feedback': 'not_helpful',
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.feedback_data.append(feedback)
            st.info("We'll use this to improve our system.")
    
    with col3:
        if st.button("🤔 Uncertain", key=f"uncertain_{result_index}"):
            feedback = {
                'result_index': result_index,
                'feedback': 'uncertain',
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.feedback_data.append(feedback)
            st.info("Thanks for letting us know.")

def render_analytics_dashboard():
    """Render analytics dashboard for fact-check history"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.fact_check_history:
        st.info("No fact-check history available. Start checking some claims!")
        return
    
    # Convert history to DataFrame for analysis
    history_data = []
    for entry in st.session_state.fact_check_history:
        verdict = entry.get('overall_verdict', '🤷‍♂️ Unverifiable')
        confidence = entry.get('overall_confidence', 0.0)
        claims_count = entry.get('claims_extracted', 0)
        
        # Extract verdict category
        if '✅' in verdict:
            verdict_category = 'True'
        elif '❌' in verdict:
            verdict_category = 'False'
        else:
            verdict_category = 'Unverifiable'
        
        history_data.append({
            'verdict': verdict_category,
            'confidence': confidence,
            'claims_extracted': claims_count,
            'timestamp': entry.get('timestamp', '')
        })
    
    df = pd.DataFrame(history_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Verdict distribution pie chart
        verdict_counts = df['verdict'].value_counts()
        fig = px.pie(
            values=verdict_counts.values,
            names=verdict_counts.index,
            title="Verdict Distribution",
            color_discrete_map={
                'True': '#28a745',
                'False': '#dc3545',
                'Unverifiable': '#ffc107'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution histogram
        fig = px.histogram(
            df, 
            x='confidence', 
            title="Confidence Score Distribution",
            nbins=20,
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Checks", len(df))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        true_percentage = (verdict_counts.get('True', 0) / len(df)) * 100
        st.metric("True Rate", f"{true_percentage:.1f}%")
    
    with col4:
        avg_claims = df['claims_extracted'].mean()
        st.metric("Avg Claims/Text", f"{avg_claims:.1f}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 LLM Fact Checker</h1>
        <p>AI-Powered Claim Verification System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Loading fact-checking pipeline..."):
            st.session_state.pipeline = load_pipeline()
            st.session_state.pipeline.setup_database(update_facts=False)
        st.success("Pipeline loaded successfully!")
    
    # Sidebar
    setup_database_ui()
    setup_configuration_ui()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Fact Check", "🔍 Batch Check", "📊 Analytics", "🔧 Tools"])
    
    with tab1:
        st.header("Single Claim Fact-Checking")
        
        # Input section
        st.markdown("### 📝 Enter your claim or statement:")
        input_text = st.text_area(
            "Text to fact-check:",
            placeholder="Example: The Indian government has announced free electricity to all farmers starting July 2025.",
            height=100
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            check_all_claims = st.checkbox("Extract and check all claims (not just primary)")
        
        with col2:
            fact_check_button = st.button("🚀 Fact Check", type="primary")
        
        # Fact-checking process
        if fact_check_button and input_text.strip():
            with st.spinner("Analyzing claim and searching for evidence..."):
                result = st.session_state.pipeline.fact_check_text(
                    input_text, 
                    extract_all_claims=check_all_claims
                )
                
                # Add to history
                st.session_state.fact_check_history.append(result)
            
            # Display results
            st.markdown("### 🎯 Fact-Check Results")
            render_verdict_card(result)
            
            # Feedback section
            render_feedback_section(len(st.session_state.fact_check_history) - 1)
    
    with tab2:
        st.header("Batch Fact-Checking")
        
        st.markdown("### 📋 Enter multiple claims (one per line):")
        batch_text = st.text_area(
            "Claims to fact-check:",
            placeholder="Claim 1: The government announced new tax reforms.\nClaim 2: GDP growth reached 8% last quarter.\nClaim 3: New vaccination program started in rural areas.",
            height=150
        )
        
        if st.button("🚀 Batch Fact Check", type="primary"):
            if batch_text.strip():
                claims = [claim.strip() for claim in batch_text.split('\n') if claim.strip()]
                
                if claims:
                    with st.spinner(f"Fact-checking {len(claims)} claims..."):
                        results = st.session_state.pipeline.batch_fact_check(claims)
                        
                        # Add to history
                        st.session_state.fact_check_history.extend(results)
                    
                    st.markdown("### 🎯 Batch Results")
                    
                    for i, result in enumerate(results):
                        st.markdown(f"#### Claim {i+1}")
                        render_verdict_card(result)
                        render_feedback_section(len(st.session_state.fact_check_history) - len(results) + i)
                        st.markdown("---")
    
    with tab3:
        render_analytics_dashboard()
    
    with tab4:
        st.header("🔧 System Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Search Facts Database")
            search_query = st.text_input("Search for facts:")
            
            if search_query:
                results = st.session_state.pipeline.search_facts(search_query, top_k=5)
                
                if results:
                    for i, fact in enumerate(results, 1):
                        with st.expander(f"Result {i} - {fact.get('source', 'Unknown')} (Score: {fact.get('similarity_score', 0):.2f})"):
                            st.write(fact.get('content', ''))
                else:
                    st.info("No matching facts found.")
        
        with col2:
            st.subheader("➕ Add Custom Facts")
            
            custom_fact = st.text_area("Enter a custom fact:")
            custom_source = st.text_input("Source:", value="Manual Entry")
            
            if st.button("Add Fact"):
                if custom_fact.strip():
                    fact_data = [{
                        'content': custom_fact,
                        'source': custom_source,
                        'title': custom_fact[:50] + "..." if len(custom_fact) > 50 else custom_fact
                    }]
                    
                    st.session_state.pipeline.add_custom_facts(fact_data)
                    st.success("Custom fact added successfully!")
        
        # Export/Import section
        st.subheader("💾 Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export History"):
                if st.session_state.fact_check_history:
                    export_data = {
                        'fact_check_history': st.session_state.fact_check_history,
                        'feedback_data': st.session_state.feedback_data,
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    # Convert to JSON for download
                    json_data = json.dumps(export_data, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        file_name=f"fact_check_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.fact_check_history = []
                st.session_state.feedback_data = []
                st.success("History cleared!")
        
        with col3:
            if st.button("🔄 Reset System"):
                st.session_state.pipeline.clear_cache()
                st.success("System cache cleared!")

if __name__ == "__main__":
    main() 