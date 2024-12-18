import streamlit as st
import pandas as pd
from database import DatabaseManager
from document_processor import DocumentProcessor
import logging
import time
import json
import uuid

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

def create_startup_form(db: DatabaseManager):
    """Show create startup form"""
    with st.form("create_startup_form"):
        st.subheader("Create New Startup")
        new_name = st.text_input("Startup Name")
        
        col1, col2 = st.columns(2)
        with col1:
            new_industry = st.text_input(
                "Industry",
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
            new_stage = st.selectbox("Stage", stage_options)
            
            new_location = st.text_input(
                "Location",
                placeholder="e.g., San Francisco, CA"
            )
        
        with col2:
            new_pitch = st.text_area(
                "One-Sentence Pitch",
                help="Describe what your startup does (up to 400 characters)",
                max_chars=400,
                height=200,
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
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Please enter a startup name")

def show_document_section(db: DatabaseManager, startup_id: str):
    """Display document management section"""
    st.subheader("Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "json"],
        help="""Supported formats:
• PDF: Pitch decks, presentations
• TXT: Meeting transcripts, notes
• JSON: Structured data (see guide below)""",
        key=f"file_uploader_{startup_id}"
    )
    
    # Document type selector
    doc_type = st.selectbox(
        "Document Type",
        options=["pitch_deck", "competitor_analysis", "market_research", "meeting_notes"],
        key=f"doc_type_{startup_id}"
    )
    
    if uploaded_file is not None:
        try:
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
    
    # Display existing documents
    st.subheader("Existing Documents")
    docs = db.get_documents(startup_id=startup_id)
    if docs:
        # Document grid
        cols = st.columns(3)
        for idx, doc in enumerate(docs):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                    #### 📄 {doc["name"]}
                    **Type:** {doc["type"]}  
                    **Uploaded:** {doc["created_at"][:10]}
                    """)
                    if st.button("🗑️ Delete", key=f"delete_{doc['id']}_{startup_id}"):
                        if db.delete_document(doc['id']):
                            st.success(f"Deleted {doc['name']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
    else:
        st.info("No documents uploaded yet")

def main():
    st.set_page_config(page_title="Startup Manager", page_icon="🚀", layout="wide")
    
    # Initialize database
    db = DatabaseManager()
    
    # Get all startups
    startups = db.get_startups()
    
    # Sidebar
    with st.sidebar:
        st.title("🚀 Navigation")
        
        # Create new startup button
        if st.button("➕ Create New Startup", use_container_width=True):
            st.session_state.show_create_startup = True
        
        st.divider()
        
        # Startup selection
        if startups:
            st.subheader("Select Startup")
            startup_names = [s['name'] for s in startups]
            selected_name = st.selectbox(
                "Choose a startup",
                startup_names,
                key='startup_select'
            )
        else:
            st.warning("No startups found. Create one first!")
            selected_name = None
    
    # Main content
    st.title("🚀 Startup Manager")
    st.caption("Manage your startup information and documents")
    
    # Show create startup form if button was clicked
    if getattr(st.session_state, 'show_create_startup', False):
        create_startup_form(db)
        if st.button("Cancel"):
            st.session_state.show_create_startup = False
            st.rerun()
    
    # Main startup management section
    if selected_name:
        # Get the selected startup's data
        selected_startup = next(s for s in startups if s['name'] == selected_name)
        
        # Startup Information Form
        st.header("Startup Information")
        with st.form(key=f'startup_form_{selected_startup["id"]}'):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "Startup Name",
                    value=selected_startup['name'],
                    disabled=True,
                    key=f"name_{selected_startup['id']}"
                )
                
                industry = st.text_input(
                    "Industry",
                    value=selected_startup.get('industry', ''),
                    placeholder="e.g., AI/ML, Healthcare, Fintech",
                    key=f"industry_{selected_startup['id']}"
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
                    index=stage_options.index(selected_startup.get('stage', 'Not specified')),
                    key=f"stage_{selected_startup['id']}"
                )
                
                location = st.text_input(
                    "Location",
                    value=selected_startup.get('location', ''),
                    placeholder="e.g., San Francisco, CA",
                    key=f"location_{selected_startup['id']}"
                )
            
            with col2:
                pitch = st.text_area(
                    "One-Sentence Pitch",
                    value=selected_startup.get('pitch', ''),
                    height=200,
                    placeholder="e.g., We provide AI-powered analytics for small businesses",
                    key=f"pitch_{selected_startup['id']}"
                )
            
            # Submit button
            submit = st.form_submit_button("💾 Save Changes")
            
            if submit:
                try:
                    # Update the startup info
                    updated_data = {
                        'pitch': pitch,
                        'industry': industry,
                        'stage': stage,
                        'location': location
                    }
                    
                    # Log the update attempt
                    logger.info(f"Updating startup {selected_startup['id']} with data: {updated_data}")
                    
                    # Perform the update
                    db.update_startup_info(selected_startup['id'], updated_data)
                    
                    # Show detailed success message
                    st.success("Changes saved successfully!")
                    
                    # Show which fields were updated
                    changes = []
                    if industry != selected_startup.get('industry', ''): changes.append("Industry")
                    if stage != selected_startup.get('stage', ''): changes.append("Stage")
                    if location != selected_startup.get('location', ''): changes.append("Location")
                    if pitch != selected_startup.get('pitch', ''): changes.append("Pitch")
                    
                    if changes:
                        st.info(f"Updated fields: {', '.join(changes)}")
                    
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving changes: {str(e)}")
                    logger.error(f"Error updating startup: {str(e)}")
        
        # Document Management Section
        st.divider()
        show_document_section(db, selected_startup['id'])
        
        # Show JSON Structure Guide
        st.divider()
        show_json_guide()

if __name__ == "__main__":
    main()
