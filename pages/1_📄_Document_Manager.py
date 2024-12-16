import streamlit as st
import pandas as pd
from database import DatabaseManager
from document_processor import DocumentProcessor
from typing import Dict, List
import json
import uuid

def show_json_guide():
    st.markdown("""
    ### JSON File Structure Guide
    When uploading JSON files, please follow these structures:
    """)
    
    structures = DatabaseManager.get_recommended_json_structure()
    for doc_type, structure in structures.items():
        with st.expander(f"{doc_type.replace('_', ' ').title()} Structure"):
            st.code(json.dumps(structure, indent=2), language="json")

def main():
    st.set_page_config(
        page_title="Document Manager",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize database connection
    db = DatabaseManager()
    
    # Sidebar navigation
    with st.sidebar:
        # Empty space to push startup selection to bottom
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] > div {
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                .startup-section {
                    margin-top: auto;
                    padding: 20px 0;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Add empty space
        st.empty()
        st.empty()
        st.empty()
        
        # Startup selection at the bottom
        st.markdown('<div class="startup-section">', unsafe_allow_html=True)
        st.subheader("Startup Selection")
        startups = db.get_startups()
        if startups:
            startup_names = [s['name'] for s in startups]
            selected_startup_name = st.selectbox("Select Startup", startup_names, label_visibility="collapsed")
        else:
            st.warning("No startups found. Create one first!")
            return
        
        # Create new startup button below the selection
        if st.button("Create New Startup"):
            st.session_state.show_create_startup = True
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Get selected startup data
    if startups:
        selected_startup = next(s for s in startups if s['name'] == selected_startup_name)
    else:
        return
    
    # Main content area
    if selected_startup:
        # File uploader with guidelines in the tooltip
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "json"],
            help="""Supported formats:
• PDF: Pitch decks, presentations
• TXT: Meeting transcripts, notes
• JSON: Structured data (see guide below)

JSON Structure Guide:
{
    "title": "Document Title",
    "type": "analysis_type",
    "content": {
        "key_findings": [...],
        "metrics": {...},
        "recommendations": [...]
    }
}""",
            key="file_uploader"
        )
        
        # Document type selector
        doc_type = st.selectbox(
            "Document Type",
            options=["pitch_deck", "competitor_analysis", "market_research", "meeting_notes"],
            key="doc_type"
        )
        
        if uploaded_file is not None:
            try:
                # Process the uploaded file
                file_content = uploaded_file.read()
                
                if db.upload_document(
                    name=uploaded_file.name,
                    content=file_content,
                    doc_type=doc_type,
                    startup_id=selected_startup['id']
                ):
                    st.success(f"Successfully uploaded {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error("Failed to upload document")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Add spacing between sections
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Display existing documents
        st.header("Existing Documents")
        docs = db.get_documents(startup_id=selected_startup['id'])
        if docs:
            # Custom CSS for better card styling
            st.markdown("""
            <style>
            .document-card {
                border: 1px solid rgba(250, 250, 250, 0.2);
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                background-color: rgba(49, 51, 63, 0.4);
                transition: all 0.3s ease;
            }
            .document-card:hover {
                border-color: rgba(250, 250, 250, 0.4);
                background-color: rgba(49, 51, 63, 0.6);
            }
            .document-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #ffffff;
            }
            .document-meta {
                font-size: 0.9rem;
                color: rgba(250, 250, 250, 0.7);
                margin-bottom: 1rem;
            }
            .delete-button {
                color: #ff6b6b;
                cursor: pointer;
                float: right;
                transition: color 0.3s ease;
            }
            .delete-button:hover {
                color: #ff4757;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a grid layout for documents
            cols = st.columns(3)  # 3 cards per row
            for idx, doc in enumerate(docs):
                with cols[idx % 3]:
                    # Icon based on document type
                    icon = "📄"
                    if doc['type'] == 'pitch_deck':
                        icon = "🎯"
                    elif doc['type'] == 'competitor_analysis':
                        icon = "📊"
                    elif doc['type'] == 'market_research':
                        icon = "🔍"
                    
                    st.markdown(f"""
                    <div class="document-card">
                        <div class="document-title">
                            {icon} {doc["name"]}
                        </div>
                        <div class="document-meta">
                            Type: {doc["type"]}<br>
                            Uploaded: {doc["created_at"][:10]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{doc['id']}", type="secondary"):
                        if db.delete_document(doc['id']):
                            st.success(f"Deleted {doc['name']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

if __name__ == "__main__":
    main()
