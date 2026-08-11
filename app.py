import streamlit as st
import pipeline
import os
import json
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Exam Answer Extractor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    groq_key = st.text_input(
        "GROQ_API_KEY", 
        type="password",
        value=os.getenv('GROQ_API_KEY', ''),
        help="Get your API key from https://console.groq.com/keys"
    )
    datalab_key = st.text_input(
        "DATALAB_API_KEY", 
        type="password",
        value=os.getenv('DATALAB_API_KEY', ''),
        help="Get your API key from https://www.datalab.to"
    )
    
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
    
    st.divider()
    st.caption("🔒 API keys are not stored and are only used for this session")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'ocr_json' not in st.session_state:
    st.session_state.ocr_json = None
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = None

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
        st.metric("Status", "Ready to process" if not st.session_state.processing else "Processing...")
    
    # Check API keys
    groq_key = st.session_state.get('groq_key') or os.getenv('GROQ_API_KEY')
    datalab_key = st.session_state.get('datalab_key') or os.getenv('DATALAB_API_KEY')
    
    if not groq_key or not datalab_key:
        st.warning("⚠️ Please enter both API keys in the sidebar to proceed.")
        st.stop()
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_clicked = st.button(
            "🚀 Process Document", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.processing
        )
    
    if process_clicked:
        st.session_state.processing = True
        st.session_state.ocr_json = None
        st.session_state.qa_pairs = None
        
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            # Store logs
            logs = []
            
            # Set up status callback
            def update_status(msg):
                logs.append(msg)
                status_text.text(f"📌 {msg}")
                
                # Update progress based on status messages
                if "Submitting document" in msg:
                    progress_bar.progress(10)
                elif "OCR complete" in msg:
                    progress_bar.progress(40)
                elif "question paper pages" in msg:
                    progress_bar.progress(60)
                elif "canonical question" in msg:
                    progress_bar.progress(70)
                elif "answer mapping" in msg:
                    progress_bar.progress(85)
                elif "Done" in msg:
                    progress_bar.progress(100)
                
                # Show logs in real-time
                with log_container:
                    st.text_area("Processing Log", "\n".join(logs[-20:]), height=150)
            
            # Read file bytes
            file_bytes = uploaded_file.read()
            
            # Process with pipeline
            status_text.text("🔄 Starting OCR processing...")
            progress_bar.progress(5)
            
            # Call the pipeline
            ocr_json, qa_pairs = pipeline.process_pdf(
                (uploaded_file.name, file_bytes),
                status_callback=update_status
            )
            
            # Store results in session state
            st.session_state.ocr_json = ocr_json
            st.session_state.qa_pairs = qa_pairs
            
            # Success!
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Display results
            st.success(f"✅ Successfully processed! Found {len(qa_pairs)} question-answer pairs.")
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)
            st.session_state.processing = False
            st.stop()
        
        st.session_state.processing = False
    
    # Display results if available
    if st.session_state.ocr_json and st.session_state.qa_pairs:
        ocr_json = st.session_state.ocr_json
        qa_pairs = st.session_state.qa_pairs
        
        # Create tabs for results
        tab1, tab2, tab3 = st.tabs(["📝 Question-Answer Pairs", "📊 Statistics", "📄 Raw OCR"])
        
        with tab1:
            st.subheader("Extracted Questions and Answers")
            
            # Search/filter
            search = st.text_input("🔍 Search questions or answers", "")
            
            filtered_pairs = qa_pairs
            if search:
                filtered_pairs = [
                    qa for qa in qa_pairs 
                    if search.lower() in qa['question'].lower() or 
                       (qa.get('answer') and search.lower() in qa['answer'].lower())
                ]
            
            if not filtered_pairs:
                st.info("No matching Q&A pairs found.")
            
            for i, qa in enumerate(filtered_pairs, 1):
                with st.expander(
                    f"Q{i}: {qa['question'][:100]}{'...' if len(qa['question']) > 100 else ''}", 
                    expanded=(i==1 and not search)
                ):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.caption(f"**Ref:** {qa['ref']}")
                        if qa['matched']:
                            st.caption(f"**Pages:** {qa.get('start_page', 'N/A')} - {qa.get('end_page', 'N/A')}")
                            st.caption(f"**Lines:** {qa.get('start_line', 'N/A')} - {qa.get('end_line', 'N/A')}")
                        else:
                            st.caption("❌ **Not matched**")
                    with col2:
                        st.markdown("**📖 Question:**")
                        st.write(qa['question'])
                        st.markdown("**✍️ Answer:**")
                        if qa['matched']:
                            answer = qa.get('answer', '')
                            if answer:
                                st.write(answer)
                            else:
                                st.warning("Answer text is empty.")
                        else:
                            st.warning("No matching answer found for this question.")
        
        with tab2:
            st.subheader("Processing Statistics")
            
            col1, col2, col3 = st.columns(3)
            matched = sum(1 for qa in qa_pairs if qa['matched'])
            total_pages = ocr_json.get('total_pages', 0)
            
            with col1:
                st.metric("Total Questions", len(qa_pairs))
            with col2:
                st.metric("Matched Answers", f"{matched}/{len(qa_pairs)}")
            with col3:
                st.metric("Total Pages", total_pages)
            
            st.divider()
            
            # Page breakdown
            st.subheader("Page Breakdown")
            
            # Get page info from OCR
            pages = ocr_json.get('pages', [])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("OCR Pages", len(pages))
            
            # Display page samples
            if pages:
                st.subheader("Page Text Samples")
                for page in pages[:3]:  # Show first 3 pages
                    with st.expander(f"Page {page.get('page_number', 'N/A')}"):
                        text = page.get('text', '')[:500]
                        st.text(text + ('...' if len(page.get('text', '')) > 500 else ''))
        
        with tab3:
            st.subheader("Raw OCR Output")
            st.json(ocr_json)
        
        # Download buttons
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            ocr_json_str = json.dumps(ocr_json, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download OCR JSON",
                data=ocr_json_str,
                file_name=f"{Path(uploaded_file.name).stem}_ocr.json",
                mime="application/json",
                use_container_width=True
            )
        with col2:
            qa_json_str = json.dumps(qa_pairs, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Q&A Pairs",
                data=qa_json_str,
                file_name=f"{Path(uploaded_file.name).stem}_qa_pairs.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # Show placeholder when no file is uploaded
    st.info("👆 Please upload a PDF file to begin processing.")
    
    # Show features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📚 Extract Questions
        Automatically identify and extract all questions from the exam paper
        """)
    with col2:
        st.markdown("""
        ### ✍️ Match Answers
        Map each question to its corresponding answer in the student's response
        """)
    with col3:
        st.markdown("""
        ### 🔍 Search & Export
        Search through Q&A pairs and export results as JSON
        """)

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit, Groq, and Datalab OCR")
st.caption("⚠️ Processing time depends on PDF size and complexity. Large documents may take several minutes.")
