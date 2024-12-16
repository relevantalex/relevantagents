import logging
from typing import Dict, List, Optional
import streamlit as st
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize Supabase client with service role key"""
        try:
            # Initialize with service role key to bypass RLS
            self.supabase = create_client(
                st.secrets["supabase"]["url"],
                st.secrets["supabase"]["service_role_key"]  # Use service role key instead of anon key
            )
            logger.info("Successfully initialized Supabase client")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            raise

    def create_startup(self, name: str, pitch: str) -> Dict:
        """Create a new startup entry"""
        try:
            data = {
                "name": name,
                "pitch": pitch
            }
            response = self.supabase.table("startups").insert(data).execute()
            logger.info(f"Created startup with name: {name}")
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
            raise

    def upload_document(self, startup_id: str, name: str, content: str, file_path: Optional[str] = None, doc_type: Optional[str] = None) -> Dict:
        """Upload a document for a startup"""
        try:
            data = {
                "startup_id": startup_id,
                "name": name,
                "content": content,
                "file_path": file_path,
                "type": doc_type
            }
            response = self.supabase.table("documents").insert(data).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise

    def get_documents(self, startup_id: Optional[str] = None) -> List[Dict]:
        """Get documents, optionally filtered by startup_id"""
        try:
            query = self.supabase.table("documents").select("*")
            if startup_id:
                query = query.eq("startup_id", startup_id)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            raise

    def upload_file_to_storage(self, file_path: str, file_data: bytes) -> str:
        """Upload a file to Supabase storage"""
        try:
            # Try to create the bucket if it doesn't exist
            try:
                self.supabase.storage.create_bucket("documents", options={'public': True})
                logger.info("Created 'documents' bucket")
            except Exception as bucket_error:
                if "already exists" not in str(bucket_error).lower():
                    logger.error(f"Error creating bucket: {str(bucket_error)}")
                    raise

            # Check if file exists and delete it (to allow overwriting)
            try:
                self.supabase.storage.from_("documents").remove([file_path])
            except:
                pass  # File doesn't exist, which is fine

            # Upload the file
            response = self.supabase.storage.from_("documents").upload(
                file_path,
                file_data,
                {"content-type": "application/pdf"}  # Add proper content type
            )
            
            # Get and verify the public URL
            public_url = self.supabase.storage.from_("documents").get_public_url(file_path)
            logger.info(f"File uploaded successfully. Public URL: {public_url}")
            return public_url

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
            raise
