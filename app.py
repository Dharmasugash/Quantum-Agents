import streamlit as st
import pandas as pd
import os
from agentic_data_system import AgenticDataSystem
import tempfile

st.set_page_config(page_title="Agentic Data System", layout="wide")

st.title("🤖 Agentic Data System")
st.markdown("""
This autonomous AI system handles **Data Ingestion, Profiling, Cleaning, and Validation**.
Upload your dataset to start the multi-agent reasoning loop.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    quality_threshold = st.slider("Target Quality Score (%)", 50, 100, 95)
    st.info("The system will iterate until this threshold is met or max iterations are reached.")

# File uploader
uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "json", "xml", "txt", "log", "pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    if st.button("🚀 Start Agentic Processing"):
        system = AgenticDataSystem()
        
        with st.spinner("Agents are working... Ingesting, Profiling, and Cleaning..."):
            try:
                # Run the agentic process
                result = system.process_dataset(tmp_path, quality_threshold=quality_threshold)
                
                # Layout for results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Processing Report")
                    st.markdown(result["report"])
                
                with col2:
                    st.subheader("✨ Cleaned Data Preview")
                    st.dataframe(result["cleaned_df"].head(100))
                    
                    # Metrics visualization
                    st.subheader("📈 Quality Metrics")
                    m = result["metrics"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Original Score", f"{m['original_score']:.1f}%")
                    c2.metric("Final Score", f"{m['final_score']:.1f}%")
                    c3.metric("Improvement", f"+{m['improvement']:.1f}%")
                    
                    # Download button
                    csv = result["cleaned_df"].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned CSV",
                        data=csv,
                        file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv",
                        mime='text/csv',
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Cleanup temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    st.info("Please upload a file to begin.")

# Footer
st.markdown("---")
st.caption("Built with Agentic AI Multi-Agent Architecture")
