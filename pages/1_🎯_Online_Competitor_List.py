import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re
import logging
import json
from urllib.parse import urlparse
import openai
from anthropic import Anthropic
from duckduckgo_search import DDGS
import csv
from io import StringIO
from database import DatabaseManager
import os

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page and theme
st.set_page_config(
    page_title="Online Competitor List",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/venture-studio',
        'Report a bug': 'https://github.com/yourusername/venture-studio/issues',
        'About': '### Venture Studio Competitor Analysis\nVersion 1.0'
    }
)

# Initialize session state
if 'competitors' not in st.session_state:
    st.session_state.competitors = {}
if 'industries' not in st.session_state:
    st.session_state.industries = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

class AIProvider:
    def __init__(self):
        self.provider = "openai"  # Default to OpenAI
        
        # Try environment variable first, then fall back to Streamlit secrets
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")
            
        self.model = "gpt-4-turbo-preview"

    def generate_response(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a startup and industry analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI generation failed: {str(e)}")
            raise

@st.cache_data(ttl=3600, show_spinner=False)
def identify_industries(pitch: str) -> List[str]:
    """Identify potential industries based on the pitch using AI"""
    ai = AIProvider()
    prompt = f"""Based on this pitch: "{pitch}"
    Identify exactly 3 specific, relevant industries or market segments.
    Format your response as a JSON array with exactly 3 strings.
    Make each industry name specific and descriptive.
    Example: ["AI-Powered Security Analytics", "Retail Technology Solutions", "Computer Vision SaaS"]"""

    try:
        response = ai.generate_response(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.strip()
        if not cleaned_response.startswith('['):
            cleaned_response = cleaned_response[cleaned_response.find('['):]
        if not cleaned_response.endswith(']'):
            cleaned_response = cleaned_response[:cleaned_response.rfind(']')+1]
        
        industries = json.loads(cleaned_response)
        return industries[:3]
    except Exception as e:
        logger.error(f"Industry identification failed: {str(e)}")
        return ["Technology Solutions", "Software Services", "Digital Innovation"]

def find_competitors(industry: str, pitch: str) -> List[Dict]:
    """Find competitors using AI and web search"""
    ai = AIProvider()
    
    try:
        # Generate search query
        search_prompt = f"""For a startup in {industry} with this pitch: "{pitch}"
        Create a search query to find direct competitors.
        Return only the search query text, nothing else."""

        search_query = ai.generate_response(search_prompt).strip().strip('"')
        
        # Perform search
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=15))
            
            # First pass: Identify potential companies
            validation_prompt = f"""Analyze these search results:
            {json.dumps(results)}
            
            First, identify which results are actual companies with products/services (not news sites, blogs, or general websites).
            Return a JSON array of indices of valid company results.
            Example: [0, 2, 5, 8]"""
            
            valid_indices_response = ai.generate_response(validation_prompt)
            valid_indices = json.loads(valid_indices_response.strip())
            valid_results = [results[i] for i in valid_indices if i < len(results)]
            
            # Second pass: Detailed analysis of valid companies
            analysis_prompt = f"""Analyze these validated competitors in {industry}:
            {json.dumps(valid_results)}
            
            Identify the top 3 most relevant direct competitors.
            For each competitor, verify they are:
            1. Real companies (not news sites or blogs)
            2. Have actual products or services
            3. Operate in the same market segment
            
            Return a JSON array with exactly 3 companies, each containing:
            {{
                "name": "Company Name",
                "website": "company website",
                "description": "2-sentence description focusing on their product/service",
                "differentiator": "key unique selling point vs your startup"
            }}
            
            Return ONLY the JSON array, no other text."""

            competitor_analysis = ai.generate_response(analysis_prompt)
            cleaned_analysis = competitor_analysis.strip()
            if not cleaned_analysis.startswith('['):
                cleaned_analysis = cleaned_analysis[cleaned_analysis.find('['):]
            if not cleaned_analysis.endswith(']'):
                cleaned_analysis = cleaned_analysis[:cleaned_analysis.rfind(']')+1]
            
            competitors = json.loads(cleaned_analysis)
            
            # Clean URLs
            for comp in competitors:
                if comp.get('website'):
                    parsed_url = urlparse(comp['website'])
                    domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path
                    if not domain.startswith('www.'):
                        domain = f"www.{domain}"
                    comp['website'] = f"https://{domain}"
            
            return competitors[:3]
            
    except Exception as e:
        logger.error(f"Competitor search failed: {str(e)}")
        raise

def export_results(startup_name: str):
    """Export analysis results to CSV"""
    if not st.session_state.competitors:
        st.warning("No analysis results to export yet.")
        return
        
    csv_data = []
    headers = ['Industry', 'Competitor', 'Website', 'Description', 'Key Differentiator']
    
    for industry, competitors in st.session_state.competitors.items():
        for comp in competitors:
            csv_data.append([
                industry,
                comp['name'],
                comp['website'],
                comp['description'],
                comp['differentiator']
            ])
    
    # Create CSV string
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(csv_data)
    
    # Create download button
    st.download_button(
        label="📥 Export Analysis",
        data=output.getvalue(),
        file_name=f"{startup_name}_competitor_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
        use_container_width=True
    )

def render_competitor_card(competitor: Dict):
    """Render a competitor analysis card using native Streamlit components"""
    with st.container():
        st.subheader(competitor['name'])
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Website**")
            st.link_button("Visit Website", competitor['website'], use_container_width=True)
        
        with col2:
            st.markdown("**Key Differentiator**")
            st.info(competitor['differentiator'])
        
        st.markdown("**Description**")
        st.text(competitor['description'])
        st.divider()

def main():
    # Initialize database connection
    db = DatabaseManager()
    
    # Initialize session state for competitors and industries
    if 'competitors' not in st.session_state:
        st.session_state.competitors = {}
    if 'industries' not in st.session_state:
        st.session_state.industries = None
    
    # Sidebar for startup selection
    with st.sidebar:
        st.subheader("Startup Selection")
        
        # Get all startups
        startups = db.get_startups()
        if startups:
            startup_names = [s['name'] for s in startups]
            selected_startup_name = st.selectbox("Select Startup", startup_names, label_visibility="collapsed")
            st.session_state.selected_startup = selected_startup_name
        else:
            st.warning("No startups found. Create one first!")
            return

    # Get selected startup from the session state
    selected_startup_name = st.session_state.get('selected_startup')
    if not selected_startup_name:
        st.warning("Please select a startup from the sidebar first.")
        return

    # Get selected startup data
    selected_startup = next(s for s in startups if s['name'] == selected_startup_name)
    
    st.title("🎯 Online Competitor List")
    st.caption("Step 1: Analyze your market and competitors")

    # Add analyze market button
    if st.button("🔍 Analyze Market", type="primary"):
        with st.spinner("Analyzing market..."):
            try:
                # First, identify industries
                st.info("Identifying relevant industries...")
                industries = identify_industries(selected_startup.get('pitch', ''))
                st.session_state.industries = industries
                st.session_state.competitors = {}
                
                # Then find competitors for each industry
                for industry in industries:
                    with st.status(f"Finding competitors in {industry}..."):
                        competitors = find_competitors(industry, selected_startup.get('pitch', ''))
                        st.session_state.competitors[industry] = competitors
                
                st.success("Market analysis completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during market analysis: {str(e)}")

    # Results section
    if st.session_state.industries and st.session_state.competitors:
        st.divider()
        
        # Create tabs for industries
        tab_titles = st.session_state.industries
        tabs = st.tabs(tab_titles)
        
        # Handle tab content
        for i, tab in enumerate(tabs):
            with tab:
                industry = tab_titles[i]
                
                # Display competitors using native components
                if industry in st.session_state.competitors:
                    for competitor in st.session_state.competitors[industry]:
                        render_competitor_card(competitor)
        
        # Export section
        if st.session_state.competitors:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                export_results(selected_startup_name)

if __name__ == "__main__":
    main()
