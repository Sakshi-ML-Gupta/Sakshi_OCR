import streamlit as st
import pipeline
import os
import tempfile
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Exam Answer Extractor", layout="wide")

# Title and description
st.title("📄 Exam Answer Extractor")
st.markdown("""
Upload a scanned exam PDF to extract questions and answers using OCR and AI.
The system will:
1. OCR the document using Datalab (Chandra model)
2. Identify question paper pages vs answer pages
3. Extract all questions
4. Map each question to its corresponding answer
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check for API keys
    groq_key = st.text_input("GROQ_API_KEY", type="password", 
                            help="Get your API key from https://console.groq.com/keys")
    datalab_key = st.text_input("DATALAB_API_KEY", type="password",
                               help="Get your API key from https://www.datalab.to")
    
    if groq_key:
        st.session_state.groq_key = groq_key
    if datalab_key:
        st.session_state.datalab_key = datalab_key
    
    st.divider()
    st.markdown("### 📌 Instructions")
    st.markdown("""
    1. Upload a PDF of a scanned exam
    2. Ensure your API keys are set
    3. Click "Process Document"
    4. Wait for processing (may take a few minutes)
    """)

# Main area - File upload
uploaded_file = st.file_uploader(
    "📤 Upload Exam PDF",
    type=["pdf"],
    help="Upload a scanned PDF of an exam booklet"
)

# Process button
if uploaded_file is not None:
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.metric("File Size", f"{file_size:.2f} MB")
    with col3:
        st.metric("Status", "Ready to process")
    
    # Check API keys
    groq_key = st.session_state.get('groq_key') or os.getenv('GROQ_API_KEY')
    datalab_key = st.session_state.get('datalab_key') or os.getenv('DATALAB_API_KEY')
    
    if not groq_key or not datalab_key:
        st.warning("⚠️ Please enter both API keys in the sidebar to proceed.")
        st.stop()
    
    # Process button
    if st.button("🚀 Process Document", type="primary"):
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Set up status callback
            def update_status(msg):
                status_text.text(f"📌 {msg}")
                # Update progress based on status messages
                if "OCR" in msg:
                    progress_bar.progress(20)
                elif "question" in msg.lower():
                    progress_bar.progress(50)
                elif "answer mapping" in msg.lower():
                    progress_bar.progress(75)
                elif "Done" in msg:
                    progress_bar.progress(100)
            
            # Read file bytes
            file_bytes = uploaded_file.read()
            
            # Process with pipeline
            status_text.text("🔄 Starting OCR processing...")
            progress_bar.progress(10)
            
            # Call the pipeline
            ocr_json, qa_pairs = pipeline.process_pdf(
                (uploaded_file.name, file_bytes),
                status_callback=update_status
            )
            
            # Success!
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Display results
            st.success(f"✅ Successfully processed! Found {len(qa_pairs)} question-answer pairs.")
            
            # Create tabs for results
            tab1, tab2, tab3 = st.tabs(["📝 Question-Answer Pairs", "📊 Statistics", "📄 Raw OCR"])
            
            with tab1:
                st.subheader("Extracted Questions and Answers")
                for i, qa in enumerate(qa_pairs, 1):
                    with st.expander(f"Q{i}: {qa['question'][:100]}...", expanded=(i==1)):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.caption(f"Ref: {qa['ref']}")
                            if qa['matched']:
                                st.caption(f"Pages: {qa['start_page']} - {qa['end_page']}")
                                st.caption(f"Lines: {qa['start_line']} - {qa['end_line']}")
                            else:
                                st.caption("❌ Not matched")
                        with col2:
                            st.markdown("**Question:**")
                            st.write(qa['question'])
                            st.markdown("**Answer:**")
                            if qa['matched']:
                                st.write(qa['answer'])
                            else:
                                st.warning("No matching answer found for this question.")
            
            with tab2:
                st.subheader("Processing Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Questions", len(qa_pairs))
                with col2:
                    matched = sum(1 for qa in qa_pairs if qa['matched'])
                    st.metric("Matched Answers", f"{matched}/{len(qa_pairs)}")
                with col3:
                    total_pages = ocr_json.get('total_pages', 0)
                    st.metric("Total Pages", total_pages)
                
                st.divider()
                st.subheader("Page Breakdown")
                st.json({
                    "Total Pages": total_pages,
                    "Question Pages": len(ocr_json.get('pages', [])),
                    "Questions Extracted": len(qa_pairs),
                    "Answers Matched": matched
                })
            
            with tab3:
                st.subheader("Raw OCR Output")
                st.json(ocr_json)
            
            # Download buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                import json
                ocr_json_str = json.dumps(ocr_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download OCR JSON",
                    data=ocr_json_str,
                    file_name=f"{Path(uploaded_file.name).stem}_ocr.json",
                    mime="application/json"
                )
            with col2:
                qa_json_str = json.dumps(qa_pairs, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download Q&A Pairs",
                    data=qa_json_str,
                    file_name=f"{Path(uploaded_file.name).stem}_qa_pairs.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)

else:
    # Show placeholder when no file is uploaded
    st.info("👆 Please upload a PDF file to begin processing.")
    st.markdown("""
    ### Example use cases:
    - 📚 Extracting questions and answers from scanned exam papers
    - 📝 Converting handwritten answer sheets to digital format
    - 🔍 Finding and organizing exam content
    """)

# Footer
st.divider()
st.caption("Built with Streamlit, Groq, and Datalab OCR")
