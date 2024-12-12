import streamlit as st
import pandas as pd
from database import DatabaseManager
from typing import Dict, List
import io
import uuid

st.set_page_config(
    page_title="Document Manager",
    page_icon="📄",
    layout="wide"
)

def process_uploaded_file(file) -> tuple[str, str]:
    """Process an uploaded file and return its content and type"""
    content = ""
    if file.type == "application/pdf":
        # Add PDF processing here
        pass
    elif file.type in ["text/plain", "text/csv"]:
        content = file.getvalue().decode("utf-8")
    
    return content, file.type

def main():
    st.title("Document Manager")
    
    db = DatabaseManager()
    
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
        # File uploader
        st.header("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "csv"],
            help="Upload pitch decks, transcripts, or other relevant documents"
        )
        
        doc_type = st.selectbox(
            "Document Type",
            ["pitch_deck", "transcript", "other"]
        )
        
        if uploaded_file and st.button("Upload Document"):
            content, file_type = process_uploaded_file(uploaded_file)
            
            # Generate unique file path
            file_path = f"{selected_startup['id']}/{str(uuid.uuid4())}/{uploaded_file.name}"
            
            # Upload to storage
            file_url = db.upload_file_to_storage(file_path, uploaded_file.getvalue())
            
            # Create document record
            doc = db.upload_document(
                startup_id=selected_startup['id'],
                file_name=uploaded_file.name,
                content=content,
                file_path=file_url,
                doc_type=doc_type
            )
            
            st.success("Document uploaded successfully!")
        
        # Display existing documents
        st.header("Existing Documents")
        docs = db.get_startup_documents(selected_startup['id'])
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['name']}"):
                    st.write(f"Type: {doc['type']}")
                    st.write(f"Uploaded: {doc['created_at']}")
                    if doc['file_path']:
                        st.link_button("View Document", doc['file_path'])
        else:
            st.info("No documents uploaded yet.")

if __name__ == "__main__":
    main()
