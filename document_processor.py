import json
from PyPDF2 import PdfReader
from io import BytesIO
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class DocumentProcessor:
    @staticmethod
    def process_file(file) -> Tuple[str, Dict]:
        """
        Process uploaded file and return content and metadata
        Returns: (content, metadata)
        """
        filename = file.name.lower()
        metadata = {
            "filename": filename,
            "file_type": file.type,
            "size": file.size
        }
        
        try:
            if filename.endswith('.txt'):
                content = file.getvalue().decode('utf-8')
                metadata["format"] = "text"
                
            elif filename.endswith('.json'):
                content = file.getvalue().decode('utf-8')
                # Validate JSON format
                try:
                    json.loads(content)
                    metadata["format"] = "json"
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON file")
                
            elif filename.endswith('.pdf'):
                pdf = PdfReader(BytesIO(file.getvalue()))
                content = ""
                for page in pdf.pages:
                    content += page.extract_text() + "\n"
                metadata["format"] = "pdf"
                metadata["pages"] = len(pdf.pages)
                
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise

    @staticmethod
    def validate_json_structure(content: str, expected_fields: list) -> bool:
        """Validate JSON content against expected structure"""
        try:
            data = json.loads(content)
            return all(field in data for field in expected_fields)
        except:
            return False

    @staticmethod
    def get_recommended_json_structure() -> Dict:
        """Return the recommended JSON structure for different document types"""
        return {
            "competitor_analysis": {
                "company_name": "string",
                "market_segment": "string",
                "target_audience": ["string"],
                "key_products": ["string"],
                "pricing_tier": "string",  # low, medium, high, enterprise
                "geographical_focus": ["string"],
                "revenue_model": "string",
                "customer_acquisition_channels": ["string"],
                "technology_stack": ["string"],
                "key_differentiators": ["string"],
                "market_positioning": {
                    "price_point": "number",  # 1-10 scale
                    "quality": "number",
                    "innovation": "number",
                    "market_share": "number"
                },
                "strengths": ["string"],
                "weaknesses": ["string"],
                "unfilled_gaps": ["string"]  # Key for niche identification
            },
            "market_research": {
                "segment": "string",
                "target_audience": ["string"],
                "pain_points": ["string"],
                "opportunities": ["string"]
            }
        }
