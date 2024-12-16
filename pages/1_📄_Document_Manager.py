import streamlit as st
import pandas as pd
from database import DatabaseManager
from document_processor import DocumentProcessor
from typing import Dict, List
import json
import uuid

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
    
    structures = DocumentProcessor.get_recommended_json_structure()
    for doc_type, structure in structures.items():
        with st.expander(f"{doc_type.replace('_', ' ').title()} Structure"):
            st.code(json.dumps(structure, indent=2), language="json")

def main():
    st.title("Document Manager")
    
    db = DatabaseManager()
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
                        expected_fields = DocumentProcessor.get_recommended_json_structure()[doc_type].keys()
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
            for doc in docs:
                with st.expander(f"📄 {doc['name']} ({doc['type']})"):
                    st.write(f"Uploaded: {doc['created_at']}")
                    if doc['content']:
                        if doc['type'] in ["competitor_analysis", "market_research"]:
                            try:
                                st.json(json.loads(doc['content']))
                            except:
                                st.text(doc['content'])
                        else:
                            st.text(doc['content'])
                    if doc['file_path']:
                        try:
                            # Ensure the file_path is a valid URL
                            download_url = doc['file_path']
                            if not download_url.startswith('http'):
                                download_url = db.supabase.storage.from_("documents").get_public_url(doc['file_path'])
                            st.markdown(f"[Download Document]({download_url}) (right-click and select 'Open in new tab' if the link doesn't work)")
                            # Also display the URL for debugging
                            st.caption(f"URL: {download_url}")
                        except Exception as e:
                            st.error(f"Error generating download link: {str(e)}")
        else:
            st.info("No documents uploaded yet.")

if __name__ == "__main__":
    main()
