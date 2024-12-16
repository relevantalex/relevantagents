import streamlit as st
import pandas as pd
from database import Database
from document_processor import DocumentProcessor
from typing import Dict, List
import json
import uuid
import fitz  # PyMuPDF

st.set_page_config(
    page_title="Document Manager",
    page_icon="📄",
    layout="wide"
)

def show_json_guide():
    st.markdown("""
    ### JSON File Structure Guide
    When uploading JSON files, please follow these structures:
    """)
    
    structures = Database.get_recommended_json_structure()
    for doc_type, structure in structures.items():
        with st.expander(f"{doc_type.replace('_', ' ').title()} Structure"):
            st.code(json.dumps(structure, indent=2), language="json")

def main():
    st.title("Document Manager")
    
    db = Database()
    doc_processor = DocumentProcessor()
    
    # Startup selector or creator
    with st.sidebar:
        st.header("Startup Selection")
        
        # Option to create new startup
        with st.expander("Create New Startup"):
            new_startup_name = st.text_input("Startup Name")
            new_startup_pitch = st.text_area("Pitch")
            if st.button("Create Startup"):
                if new_startup_name and new_startup_pitch:
                    startup = db.create_startup(new_startup_name, new_startup_pitch)
                    st.success(f"Created startup: {new_startup_name}")
                    st.rerun()
        
        # Startup selector
        startups = db.get_startups()
        if startups:
            startup_names = [s["name"] for s in startups]
            selected_startup_name = st.selectbox(
                "Select Startup",
                startup_names
            )
            selected_startup = next(s for s in startups if s["name"] == selected_startup_name)
        else:
            st.warning("No startups found. Create one first!")
            return

    # Main content area
    if selected_startup:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Upload Documents")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "txt", "json"],
                help="Upload pitch decks, transcripts, or other relevant documents"
            )
            
            doc_type = st.selectbox(
                "Document Type",
                ["pitch_deck", "transcript", "competitor_analysis", 
                 "market_research", "other"]
            )
        
        with col2:
            st.header("File Guidelines")
            st.info("""
            Supported formats:
            - PDF: Pitch decks, presentations
            - TXT: Meeting transcripts, notes
            - JSON: Structured data (see guide below)
            """)
            
            if doc_type in ["competitor_analysis", "market_research"]:
                show_json_guide()
        
        if uploaded_file and st.button("Upload Document"):
            try:
                with st.spinner("Processing document..."):
                    # Process the file
                    content, metadata = doc_processor.process_file(uploaded_file)
                    
                    # For JSON files, validate structure if needed
                    if metadata["format"] == "json" and doc_type in ["competitor_analysis", "market_research"]:
                        expected_fields = Database.get_recommended_json_structure()[doc_type].keys()
                        if not doc_processor.validate_json_structure(content, expected_fields):
                            st.error("JSON structure does not match the recommended format!")
                            return
                    
                    # Generate unique file path
                    file_path = f"{selected_startup['id']}/{str(uuid.uuid4())}/{uploaded_file.name}"
                    
                    # Upload to storage
                    file_url = db.upload_file_to_storage(file_path, uploaded_file.getvalue())
                    
                    # Create document record
                    doc = db.upload_document(
                        startup_id=selected_startup['id'],
                        name=uploaded_file.name,
                        content=content,
                        file_path=file_url,
                        doc_type=doc_type
                    )
                    
                    st.success("Document uploaded and processed successfully!")
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
        
        # Display existing documents
        st.header("Existing Documents")
        docs = db.get_documents(startup_id=selected_startup['id'])
        if docs:
            # Custom CSS for better card styling
            st.markdown("""
            <style>
            .document-card {
                border: 1px solid #2f3640;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background-color: #1e242e;
                position: relative;
                min-height: 300px;
            }
            .card-content {
                margin-bottom: 60px;  /* Space for the button */
            }
            .download-button {
                position: absolute;
                bottom: 20px;
                left: 20px;
                right: 20px;
                background-color: #4a69bd;
                color: white;
                padding: 10px;
                text-align: center;
                border-radius: 5px;
                text-decoration: none;
                display: block;
                transition: background-color 0.3s ease;
            }
            .download-button:hover {
                background-color: #6a89cc;
                color: white;
                text-decoration: none;
            }
            .document-title {
                color: #ffffff;
                margin-bottom: 10px;
            }
            .document-meta {
                color: #8395a7;
                font-size: 0.9em;
                margin-bottom: 15px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a grid layout for documents
            cols = st.columns(3)  # 3 cards per row
            for idx, doc in enumerate(docs):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown('<div class="document-card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-content">', unsafe_allow_html=True)
                        
                        # Icon based on document type
                        icon = "📄"
                        if doc['type'] == 'pitch_deck':
                            icon = "🎯"
                        elif doc['type'] == 'competitor_analysis':
                            icon = "📊"
                        elif doc['type'] == 'market_research':
                            icon = "🔍"
                        
                        # Document title and type with better styling
                        st.markdown(f'<h3 class="document-title">{icon} {doc["name"]}</h3>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="document-meta">Type: {doc["type"]}<br>Uploaded: {doc["created_at"][:10]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Preview section
                        if doc['file_path']:
                            try:
                                download_url = doc['file_path']
                                if not download_url.startswith('http'):
                                    download_url = db.supabase.storage.from_("documents").get_public_url(doc['file_path'])
                                
                                # Get PDF content
                                response = requests.get(download_url)
                                if response.status_code == 200 and doc['name'].lower().endswith('.pdf'):
                                    # Create a temporary file to store the PDF
                                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                                        temp_file.write(response.content)
                                        temp_file.flush()
                                        
                                        # Extract text from first page
                                        pdf_reader = PyPDF2.PdfReader(temp_file.name)
                                        if len(pdf_reader.pages) > 0:
                                            first_page = pdf_reader.pages[0]
                                            preview_text = first_page.extract_text()[:200]
                                            st.text_area("Preview", preview_text, height=100)
                            except Exception as e:
                                st.error(f"Error loading preview: {str(e)}")
                        
                        # Content preview if available
                        if doc['content']:
                            with st.expander("Content Preview"):
                                if doc['type'] in ["competitor_analysis", "market_research"]:
                                    try:
                                        st.json(json.loads(doc['content']))
                                    except:
                                        st.text(doc['content'][:200] + "...")
                                else:
                                    st.text(doc['content'][:200] + "...")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button at the bottom
                        if doc['file_path']:
                            try:
                                download_url = doc['file_path']
                                if not download_url.startswith('http'):
                                    download_url = db.supabase.storage.from_("documents").get_public_url(doc['file_path'])
                                st.markdown(f'<a href="{download_url}" class="download-button" target="_blank">📥 Download Document</a>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error with download link: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No documents uploaded yet.")

if __name__ == "__main__":
    main()
