import streamlit as st
import pandas as pd
from database import DatabaseManager
import logging
import time

logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="Startup Manager", page_icon="", layout="wide")
    
    # Initialize database
    db = DatabaseManager()
    
    # Get all startups
    startups = db.get_startups()
    if not startups:
        st.warning("No startups found. Create one first!")
        return

    # Main layout
    st.title("")
    
    # Startup Selection
    startup_names = [s['name'] for s in startups]
    selected_name = st.selectbox("Select Startup", startup_names, key='startup_select')
    
    # Get the selected startup's data
    selected_startup = next(s for s in startups if s['name'] == selected_name)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with st.form(key='startup_form'):
        with col1:
            # Basic Info
            name = st.text_input("Startup Name", value=selected_startup['name'], disabled=True)
            
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
        submit = st.form_submit_button("Save Changes")
        
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
                
                # Show success message
                st.success("Changes saved successfully!")
                
                # Force a page refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving changes: {str(e)}")
                logger.error(f"Error updating startup: {str(e)}")

    # Add a divider
    st.divider()
    
    # Display current values for verification
    st.subheader("Current Startup Information")
    st.json({
        'name': selected_startup['name'],
        'industry': selected_startup.get('industry', 'Not specified'),
        'stage': selected_startup.get('stage', 'Not specified'),
        'location': selected_startup.get('location', ''),
        'pitch': selected_startup.get('pitch', '')
    })

if __name__ == "__main__":
    main()
