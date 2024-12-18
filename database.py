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

    def create_startup(self, name: str, pitch: str = "", industry: str = "Not specified", stage: str = "Not specified", location: str = "") -> Dict:
        """Create a new startup entry"""
        try:
            data = {
                "name": name,
                "pitch": pitch,
                "industry": industry,
                "stage": stage,
                "location": location
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
            logger.info("Fetching all startups from database")
            response = self.supabase.table("startups").select("*").execute()
            logger.info(f"Retrieved {len(response.data)} startups")
            for startup in response.data:
                logger.info(f"Startup data: {startup}")
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

    def delete_document(self, document_id: int) -> bool:
        """Delete a document from the database."""
        try:
            response = self.supabase.table('documents').delete().eq('id', document_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def update_startup_pitch(self, startup_id: str, pitch: str) -> Dict:
        """Update a startup's pitch"""
        try:
            data = {"pitch": pitch}
            response = self.supabase.table("startups").update(data).eq("id", startup_id).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Error updating startup pitch: {str(e)}")
            raise

    def update_startup_info(self, startup_id: str, info: Dict[str, str]) -> Dict:
        """Update startup information including pitch, industry, stage, and location"""
        try:
            # For now, only update the pitch until we add the new columns
            update_data = {"pitch": info.get('pitch', '')}
            
            # Log the update attempt
            logger.info(f"Updating startup {startup_id} with pitch")
            
            # Update pitch only for now
            response = self.supabase.table("startups").update(update_data).eq("id", startup_id).execute()
            
            if not response.data:
                raise Exception("No data returned from update operation")
                
            logger.info(f"Successfully updated startup {startup_id}")
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error updating startup info: {str(e)}")
            raise Exception(f"Failed to update startup information: {str(e)}")

    @staticmethod
    def get_recommended_json_structure():
        """Get recommended JSON structure for different document types."""
        return {
            "competitor_analysis": {
                "company_name": "string",
                "website": "string",
                "products": ["string"],
                "strengths": ["string"],
                "weaknesses": ["string"],
                "opportunities": ["string"],
                "threats": ["string"]
            },
            "market_research": {
                "market_size": "number",
                "growth_rate": "number",
                "key_players": ["string"],
                "trends": ["string"],
                "challenges": ["string"]
            }
        }

    def validate_json_structure(self, content: Dict, expected_fields: List[str]) -> bool:
        """Validate JSON structure against expected fields."""
        try:
            content_fields = set(content.keys())
            expected_fields = set(expected_fields)
            return content_fields.issuperset(expected_fields)
        except:
            return False
