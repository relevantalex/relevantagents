import streamlit as st
import pandas as pd
from database import DatabaseManager
from document_processor import DocumentProcessor
from typing import Dict, List
import json
import uuid
import time
import logging

logger = logging.getLogger(__name__)

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
        page_title="🚀 Document Manager",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize database connection
    db = DatabaseManager()
    
    # Get fresh startup data
    startups = db.get_startups()
    
    # Sidebar navigation
    with st.sidebar:
        # Main navigation
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add a container for the rest of the sidebar space
        with st.container():
            # Add vertical space
            for _ in range(10):
                st.empty()
            
            # Startup selection at the bottom
            st.subheader("Startup Selection")
            
            if startups:
                startup_names = [s['name'] for s in startups]
                
                # Initialize session state if needed
                if 'selected_startup_name' not in st.session_state:
                    st.session_state.selected_startup_name = startup_names[0]
                elif st.session_state.selected_startup_name not in startup_names:
                    st.session_state.selected_startup_name = startup_names[0]
                
                selected_startup_name = st.selectbox(
                    "Select Startup",
                    startup_names,
                    index=startup_names.index(st.session_state.selected_startup_name),
                    key='startup_selector'
                )
                
                # Update session state
                if selected_startup_name != st.session_state.selected_startup_name:
                    st.session_state.selected_startup_name = selected_startup_name
                    st.session_state.selected_startup = next(s for s in startups if s['name'] == selected_startup_name)
                
                # Create new startup button below the selection
                if st.button("Create New Startup"):
                    st.session_state.show_create_startup = True
                    st.session_state.new_startup_name = ""
                    st.session_state.new_startup_pitch = ""
                    st.session_state.new_startup_industry = "Not specified"
                    st.session_state.new_startup_stage = "Not specified"
                    st.session_state.new_startup_location = ""

                # Show create startup form if button was clicked
                if getattr(st.session_state, 'show_create_startup', False):
                    with st.form("create_startup_form"):
                        st.subheader("Create New Startup")
                        new_name = st.text_input("Startup Name", key="new_startup_name")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            new_industry = st.text_input(
                                "Industry",
                                key="new_startup_industry",
                                placeholder="e.g., AI/ML, Healthcare, Fintech"
                            )
                            
                            stage_options = [
                                "Not specified",
                                "Pre-seed",
                                "Seed",
                                "Series A",
                                "Series B",
                                "Series C",
                                "Series D+",
                                "Growth"
                            ]
                            new_stage = st.selectbox(
                                "Stage",
                                stage_options,
                                key="new_startup_stage"
                            )
                            
                            new_location = st.text_input(
                                "Location",
                                key="new_startup_location",
                                placeholder="e.g., San Francisco, CA"
                            )
                        
                        with col2:
                            new_pitch = st.text_area(
                                "One-Sentence Pitch", 
                                key="new_startup_pitch", 
                                help="Describe what your startup does (up to 400 characters)",
                                max_chars=400,
                                height=300,
                                placeholder="e.g., We provide AI-powered analytics for small businesses"
                            )
                        
                        submitted = st.form_submit_button("Create")
                        if submitted:
                            if new_name:
                                startup = db.create_startup(
                                    name=new_name,
                                    pitch=new_pitch,
                                    industry=new_industry,
                                    stage=new_stage,
                                    location=new_location
                                )
                                st.success(f"Created startup: {new_name}")
                                st.session_state.show_create_startup = False
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Please enter a startup name")
            else:
                st.warning("No startups found. Create one first!")
                return
    
    # Get selected startup data
    if startups:
        # Get fresh startup data every time
        selected_startup = next(s for s in startups if s['name'] == st.session_state.selected_startup_name)
        startup_id = selected_startup['id']
        
        # Main content area
        st.title("🚀 Startup Manager")
        st.caption("Step 2: Manage your startup information and documents")
        
        # Startup Information Section
        with st.container():
            st.subheader("Startup Information")
            
            # Name and Pitch in two columns
            col1, col2 = st.columns(2)
            with col1:
                startup_name = st.text_input(
                    "Startup Name", 
                    value=selected_startup['name'],
                    disabled=True
                )
                
                industry = st.text_input(
                    "Industry",
                    value=selected_startup.get('industry', ''),
                    placeholder="e.g., AI/ML, Healthcare, Fintech"
                )
                
                stage_options = [
                    "Not specified",
                    "Pre-seed",
                    "Seed",
                    "Series A",
                    "Series B",
                    "Series C",
                    "Series D+",
                    "Growth"
                ]
                stage = st.selectbox(
                    "Stage",
                    stage_options,
                    index=stage_options.index(selected_startup.get('stage', 'Not specified'))
                )
                
                location = st.text_input(
                    "Location",
                    value=selected_startup.get('location', ''),
                    placeholder="e.g., San Francisco, CA"
                )
            
            with col2:
                pitch = st.text_area(
                    "One-Sentence Pitch",
                    value=selected_startup.get('pitch', ''),
                    help="Describe what your startup does (up to 400 characters)",
                    max_chars=400,
                    height=300,
                    placeholder="e.g., We provide AI-powered analytics for small businesses"
                )
                
                # Save all fields button
                if st.button("Save Changes"):
                    try:
                        # Save current startup name for reselection
                        current_startup_name = selected_startup['name']
                        
                        # Update all fields in one operation
                        updated_startup = db.update_startup_info(
                            startup_id,
                            {
                                'pitch': pitch,
                                'industry': industry,
                                'stage': stage,
                                'location': location
                            }
                        )
                        
                        if updated_startup:
                            st.success("Changes saved successfully!")
                            
                            # Store the current startup name in session state
                            st.session_state.selected_startup_name = current_startup_name
                            
                            # Show which fields were updated
                            if industry != selected_startup.get('industry', ''):
                                st.info("Industry updated")
                            if stage != selected_startup.get('stage', ''):
                                st.info("Stage updated")
                            if location != selected_startup.get('location', ''):
                                st.info("Location updated")
                            if pitch != selected_startup.get('pitch', ''):
                                st.info("Pitch updated")
                            
                            # Use a shorter delay
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error saving changes: {str(e)}")
                        logger.error(f"Error saving startup changes: {str(e)}")
    
    st.divider()
    
    # Document Management Section
    st.subheader("Document Management")
    
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
                startup_id=startup_id
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
    docs = db.get_documents(startup_id=startup_id)
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
