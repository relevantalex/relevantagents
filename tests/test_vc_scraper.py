import os
import asyncio
import aiohttp
import ssl
import certifi
import json
from typing import List, Dict
import logging
import sys
from pathlib import Path
from mock_database import MockDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API keys for testing
os.environ['GOOGLE_API_KEY'] = 'dummy_google_api_key'
os.environ['GOOGLE_CX'] = 'dummy_google_cx'
os.environ['OPENAI_API_KEY'] = 'dummy_openai_api_key'

class MockOpenAIResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]

class MockChatCompletion:
    @classmethod
    async def create(cls, model, messages, temperature=0.7, max_tokens=500):
        """Mock chat completion create method"""
        return MockOpenAIResponse(json.dumps({
            'firm_name': 'Test VC Firm',
            'investment_focus': 'Healthcare technology and biotech startups',
            'stage_preference': 'Seed and early stage',
            'relevance_score': 0.9,
            'contact_information': {
                'emails': ['contact@testvc.com'],
                'linkedin_profiles': ['https://linkedin.com/company/testvc']
            }
        }))
    
    @classmethod
    async def acreate(cls, model, messages, temperature=0.7, max_tokens=500):
        """Mock async chat completion create method"""
        return MockOpenAIResponse(json.dumps({
            'firm_name': 'Test VC Firm',
            'investment_focus': 'Healthcare technology and biotech startups',
            'stage_preference': 'Seed and early stage',
            'relevance_score': 0.9,
            'contact_information': {
                'emails': ['contact@testvc.com'],
                'linkedin_profiles': ['https://linkedin.com/company/testvc']
            }
        }))

class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = type('Chat', (), {
            'completions': type('Completions', (), {
                'create': MockChatCompletion.create,
                'acreate': MockChatCompletion.acreate
            })()
        })

class MockStreamlit:
    """Mock Streamlit functions for testing"""
    @staticmethod
    def write(*args, **kwargs):
        logger.info(*args)
    
    @staticmethod
    def error(*args, **kwargs):
        logger.error(*args)
    
    @staticmethod
    def warning(*args, **kwargs):
        logger.warning(*args)
    
    @staticmethod
    def success(*args, **kwargs):
        logger.info(*args)
    
    @staticmethod
    def progress(*args, **kwargs):
        class MockProgress:
            def progress(self, *args, **kwargs):
                pass
        return MockProgress()
    
    @staticmethod
    def spinner(*args, **kwargs):
        class MockSpinner:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return MockSpinner()
    
    @staticmethod
    def secrets():
        return {
            'api_keys': {
                'google_api_key': os.environ['GOOGLE_API_KEY'],
                'google_cx': os.environ['GOOGLE_CX'],
                'openai_api_key': os.environ['OPENAI_API_KEY']
            },
            'model_settings': {
                'openai_model': 'gpt-4-1106-preview',
                'temperature': 0.7,
                'max_tokens': 500,
                'enrichment_model': 'gpt-4-1106-preview',
                'enrichment_temperature': 0.7,
                'enrichment_max_tokens': 1000
            }
        }

# Replace streamlit with mock for testing
import streamlit as st
st.write = MockStreamlit.write
st.error = MockStreamlit.error
st.warning = MockStreamlit.warning
st.success = MockStreamlit.success
st.progress = MockStreamlit.progress
st.spinner = MockStreamlit.spinner
st.secrets = MockStreamlit.secrets()

# Mock the database module
sys.modules['database'] = type('MockDatabase', (), {
    'DatabaseManager': MockDatabaseManager
})

# Mock OpenAI module
sys.modules['openai'] = type('MockOpenAI', (), {
    'OpenAI': MockOpenAIClient,
    'AsyncOpenAI': MockOpenAIClient,
    'ChatCompletion': MockChatCompletion
})

# Mock aiohttp session to include referrer
class MockClientSession(aiohttp.ClientSession):
    def __init__(self, *args, **kwargs):
        headers = kwargs.get('headers', {})
        headers.update({
            'Referer': 'http://localhost:8501',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        kwargs['headers'] = headers
        super().__init__(*args, **kwargs)

# Replace aiohttp.ClientSession with our mock
aiohttp.ClientSession = MockClientSession

# Import VCSearchAgent using importlib to handle emoji in filename
vc_scraper_path = str(Path(__file__).parent.parent / "pages" / "2_🔍_Relevant_VC_Scraper.py")
import importlib.util
spec = importlib.util.spec_from_file_location("vc_scraper", vc_scraper_path)
vc_scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc_scraper)
VCSearchAgent = vc_scraper.VCSearchAgent

async def test_vc_search():
    """Test the VC search functionality"""
    agent = VCSearchAgent()
    
    # Test search parameters
    industry = "healthcare technology"
    stage = "seed"
    max_results = 5
    
    try:
        logger.info("Initializing VCSearchAgent...")
        logger.info("\nStarting VC search...")
        logger.info(f"Industry: {industry}")
        logger.info(f"Stage: {stage}")
        logger.info(f"Max Results: {max_results}")
        
        results = await agent.search(industry, stage, max_results)
        
        logger.info(f"Search completed. Found {len(results)} results.")
        
        if results:
            logger.info("\nSearch Results:")
            for i, result in enumerate(results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Name: {result.name}")
                logger.info(f"URL: {result.url}")
                logger.info(f"Description: {result.description[:200]}...")
                logger.info("-" * 80)
            
            return True
        else:
            logger.error("No results found.")
            return False
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_vc_search())
    if success:
        print("\n✅ VC search test completed successfully!")
    else:
        print("\n❌ VC search test failed!")
        exit(1)
