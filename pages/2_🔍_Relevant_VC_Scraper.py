import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
from urllib.parse import urlparse
import os
from database import DatabaseManager
import time
import json
from duckduckgo_search import DDGS
import openai
from anthropic import Anthropic
import random
import asyncio
import aiohttp
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = "Waiting"
        self.progress = 0
        self.messages = []
        self.start_time = None
        
    def start(self):
        self.status = "Running"
        self.start_time = datetime.now()
        
    def complete(self):
        self.status = "Complete"
        self.progress = 100
        
    def fail(self, error: str):
        self.status = "Failed"
        self.messages.append({"type": "error", "content": error})
        
    def update_progress(self, progress: int, message: str = None):
        self.progress = progress
        if message:
            self.messages.append({"type": "info", "content": message})
            
    def get_runtime(self) -> str:
        if not self.start_time:
            return "Not started"
        delta = datetime.now() - self.start_time
        return f"{delta.seconds}s"

class IndustryAnalysisAgent:
    def __init__(self):
        self.status = AgentStatus(
            "Industry Analysis Agent",
            "Analyzing industry landscape and identifying key trends"
        )
        # Initialize OpenAI client with API key
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    async def analyze(self, startup_data: Dict[str, str]) -> Dict:
        self.status.start()
        try:
            # Step 1: Extract key industry terms
            self.status.update_progress(20, "Extracting key industry terms...")
            industry_terms = await self._extract_industry_terms(startup_data)
            
            # Step 2: Research industry trends
            self.status.update_progress(40, "Researching industry trends...")
            industry_trends = await self._research_trends(industry_terms)
            
            # Step 3: Identify key players
            self.status.update_progress(60, "Identifying key players...")
            key_players = await self._identify_key_players(industry_terms)
            
            # Step 4: Analyze market dynamics
            self.status.update_progress(80, "Analyzing market dynamics...")
            market_dynamics = await self._analyze_market(industry_terms, key_players)
            
            # Complete analysis
            self.status.complete()
            
            return {
                "industry_terms": industry_terms,
                "trends": industry_trends,
                "key_players": key_players,
                "market_dynamics": market_dynamics
            }
            
        except Exception as e:
            self.status.fail(str(e))
            raise
    
    async def _extract_industry_terms(self, startup_data: Dict[str, str]) -> List[str]:
        prompt = f"""You are an AI trained to analyze startup information and extract relevant industry terms.
        
        Based on the following startup information, provide a JSON array of relevant industry terms and categories.
        The response should be ONLY a valid JSON array of strings, nothing else.

        Startup Information:
        Industry: {startup_data['industry']}
        Pitch: {startup_data['pitch']}
        Stage: {startup_data['stage']}
        
        Example response format:
        ["term1", "term2", "term3"]
        """
        
        try:
            response = await self._get_gpt4_response(prompt)
            # Clean the response and ensure it's valid JSON
            response = response.strip()
            if not response.startswith('['):
                response = '[' + response
            if not response.endswith(']'):
                response = response + ']'
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT-4 response: {response}")
            # Return a basic array with the industry if parsing fails
            return [startup_data['industry']]
            
    async def _research_trends(self, industry_terms: List[str]) -> List[Dict]:
        trends = []
        ddgs = DDGS()
        
        for term in industry_terms[:3]:  # Limit to top 3 terms
            try:
                results = ddgs.news(term, max_results=5)
                for result in results:
                    trends.append({
                        "term": term,
                        "title": result['title'],
                        "snippet": result['body'],
                        "source": result['source'],
                        "date": result['date']
                    })
            except Exception as e:
                logger.error(f"Error researching trends for {term}: {str(e)}")
        
        return trends
    
    async def _identify_key_players(self, industry_terms: List[str]) -> List[Dict]:
        prompt = f"""You are an AI trained to identify key companies in specific industries.
        
        For these industry terms: {', '.join(industry_terms)}
        
        Return a JSON array of objects with the top companies and their value propositions.
        The response should be ONLY a valid JSON array, nothing else.
        
        Example response format:
        [
            {{"name": "Company1", "value_prop": "Description1"}},
            {{"name": "Company2", "value_prop": "Description2"}}
        ]
        """
        
        try:
            response = await self._get_gpt4_response(prompt)
            # Clean the response and ensure it's valid JSON
            response = response.strip()
            if not response.startswith('['):
                response = '[' + response
            if not response.endswith(']'):
                response = response + ']'
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT-4 response: {response}")
            return [{"name": "Analysis Failed", "value_prop": "Could not parse key players"}]
    
    async def _analyze_market(self, industry_terms: List[str], key_players: List[Dict]) -> Dict:
        prompt = f"""You are an AI trained to analyze market dynamics.
        
        Analyze the market for:
        Industry Terms: {', '.join(industry_terms)}
        Key Players: {json.dumps(key_players)}
        
        Return a JSON object with market analysis.
        The response should be ONLY a valid JSON object, nothing else.
        
        Example response format:
        {{
            "market_size": "Size description",
            "growth_rate": "Growth description",
            "key_challenges": ["Challenge 1", "Challenge 2"],
            "opportunities": ["Opportunity 1", "Opportunity 2"]
        }}
        """
        
        try:
            response = await self._get_gpt4_response(prompt)
            # Clean the response and ensure it's valid JSON
            response = response.strip()
            if not response.startswith('{'):
                response = '{' + response
            if not response.endswith('}'):
                response = response + '}'
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT-4 response: {response}")
            return {
                "market_size": "Analysis Failed",
                "growth_rate": "Analysis Failed",
                "key_challenges": ["Could not parse market analysis"],
                "opportunities": ["Could not parse market analysis"]
            }
    
    async def _get_gpt4_response(self, prompt: str) -> str:
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert industry analyst AI. Always respond with valid JSON only, no additional text."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting GPT-4 response: {str(e)}")
            raise

def display_agent_status(agent_status: AgentStatus):
    """Display a cool agent status interface"""
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            status_color = {
                "Waiting": "⚪️",
                "Running": "🔵",
                "Complete": "🟢",
                "Failed": "🔴"
            }
            st.write(f"{status_color[agent_status.status]} **{agent_status.name}**")
        
        with col2:
            st.progress(agent_status.progress)
        
        with col3:
            st.write(f"⏱️ {agent_status.get_runtime()}")
        
        # Display messages with cool formatting
        for msg in agent_status.messages[-3:]:  # Show last 3 messages
            if msg["type"] == "error":
                st.error(msg["content"])
            else:
                st.info(msg["content"])

async def find_relevant_vcs(startup_data: Dict[str, any]):
    """Find relevant VCs through multi-agent analysis"""
    
    # Initialize agents
    industry_agent = IndustryAnalysisAgent()
    
    # Create status container
    status_container = st.empty()
    
    try:
        # Run Industry Analysis
        industry_analysis = await industry_agent.analyze(startup_data)
        
        # Update status display
        with status_container:
            display_agent_status(industry_agent.status)
        
        if industry_agent.status.status == "Complete":
            # Display industry analysis results
            st.subheader("🎯 Industry Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Industry Terms:**")
                for term in industry_analysis["industry_terms"]:
                    st.write(f"- {term}")
                
                st.write("**Key Players:**")
                for player in industry_analysis["key_players"]:
                    st.write(f"- **{player['name']}**: {player['value_prop']}")
            
            with col2:
                st.write("**Recent Trends:**")
                for trend in industry_analysis["trends"][:3]:
                    st.write(f"- **{trend['title']}**")
                    st.caption(f"Source: {trend['source']} ({trend['date']})")
            
            st.write("**Market Dynamics:**")
            market = industry_analysis["market_dynamics"]
            st.write(f"- Market Size: {market['market_size']}")
            st.write(f"- Growth Rate: {market['growth_rate']}")
            st.write("- Key Challenges:")
            for challenge in market['key_challenges']:
                st.write(f"  - {challenge}")
            
            # Continue with next phase...
            st.info("Industry analysis complete! Ready for VC matching phase...")
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {str(e)}")

def load_startup_data() -> Dict[str, any]:
    """Load startup data from the database"""
    db = DatabaseManager()
    
    # Get all startups
    startups = db.get_startups()
    if not startups:
        st.error("No startups found. Please create a startup in the Startup Manager first.")
        return None
    
    # Get selected startup from session state or select the first one
    if 'selected_startup' not in st.session_state:
        st.session_state.selected_startup = startups[0]
    
    # Create startup selector in sidebar
    with st.sidebar:
        st.subheader("Startup Selection")
        startup_names = [s['name'] for s in startups]
        selected_name = st.selectbox(
            "Select Startup",
            startup_names,
            index=startup_names.index(st.session_state.selected_startup['name'])
        )
        
        # Update selected startup in session state
        st.session_state.selected_startup = next(s for s in startups if s['name'] == selected_name)
    
    # Prepare startup data
    startup_data = {
        'name': st.session_state.selected_startup['name'],
        'pitch': st.session_state.selected_startup.get('pitch', ''),
        'industry': st.session_state.selected_startup.get('industry', 'Not specified'),
        'stage': st.session_state.selected_startup.get('stage', 'Not specified'),
        'location': st.session_state.selected_startup.get('location', 'Not specified')
    }
    
    return startup_data

def main():
    st.set_page_config(
        page_title="Relevant VC Scraper",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Relevant VC Scraper")
    st.caption("Intelligent multi-agent system for finding your perfect VC match")
    
    # Check for OpenAI API key
    if not (os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")):
        st.error("""
        ⚠️ OpenAI API key is missing!
        
        Please set your OpenAI API key in one of these ways:
        1. Add it to your environment variables as OPENAI_API_KEY
        2. Add it to your Streamlit secrets.toml file under [api_keys] openai_api_key
        """)
        return
    
    # Load startup data
    startup_data = load_startup_data()
    if not startup_data:
        return
    
    # Display current startup info
    st.subheader("Current Startup Profile")
    st.write(f"**Name:** {startup_data['name']}")
    st.write(f"**Industry:** {startup_data['industry']}")
    st.write(f"**Stage:** {startup_data['stage']}")
    st.write(f"**Location:** {startup_data['location']}")
    st.write(f"**Pitch:** {startup_data['pitch']}")
    
    # Start analysis button
    if st.button("🚀 Start Deep Analysis", use_container_width=True):
        asyncio.run(find_relevant_vcs(startup_data))

if __name__ == "__main__":
    main()
