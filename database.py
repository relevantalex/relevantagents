from supabase import create_client
import streamlit as st
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.supabase = create_client(
            st.secrets["supabase"]["url"],
            st.secrets["supabase"]["anon_key"]
        )
    
    def create_startup(self, name: str, pitch: str) -> Dict:
        """Create a new startup entry"""
        try:
            response = self.supabase.table("startups").insert({
                "name": name,
                "pitch": pitch
            }).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Error creating startup: {str(e)}")
            raise

    def get_startups(self) -> List[Dict]:
        """Get all startups"""
        try:
            response = self.supabase.table("startups").select("*").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching startups: {str(e)}")
            return []

    def upload_document(self, startup_id: str, file_name: str, content: str, 
                       file_path: str, doc_type: str) -> Dict:
        """Upload a document associated with a startup"""
        try:
            response = self.supabase.table("documents").insert({
                "startup_id": startup_id,
                "name": file_name,
                "content": content,
                "file_path": file_path,
                "type": doc_type
            }).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise

    def get_startup_documents(self, startup_id: str) -> List[Dict]:
        """Get all documents for a specific startup"""
        try:
            response = self.supabase.table("documents")\
                .select("*")\
                .eq("startup_id", startup_id)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    def upload_file_to_storage(self, file_path: str, file_data: bytes) -> str:
        """Upload a file to Supabase storage"""
        try:
            response = self.supabase.storage.from_("documents").upload(
                file_path,
                file_data
            )
            return self.supabase.storage.from_("documents").get_public_url(file_path)
        except Exception as e:
            logger.error(f"Error uploading file to storage: {str(e)}")
            raise

    def save_analysis(self, analysis_data: Dict) -> Dict:
        """Save an analysis record to the database"""
        try:
            response = self.supabase.table("analyses").insert(analysis_data).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

    def get_analyses_for_startup(self, startup_id: str, analysis_type: Optional[str] = None) -> List[Dict]:
        """Get all analyses for a startup"""
        try:
            query = self.supabase.table("analyses").select("*").eq("startup_id", startup_id)
            if analysis_type:
                query = query.eq("analysis_type", analysis_type)
            response = query.order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching analyses: {str(e)}")
            raise
