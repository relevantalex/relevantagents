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

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page and theme
st.set_page_config(
    page_title="Competitor Analysis",
    page_icon="📊",
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
        self.provider = st.secrets.get("api_settings", {}).get("ai_provider", "openai")
        
        if self.provider == "openai":
            openai.api_key = st.secrets["api_keys"]["openai_api_key"]
            self.model = "gpt-4-turbo-preview"
        else:
            self.anthropic = Anthropic(api_key=st.secrets["api_keys"]["anthropic_api_key"])
            self.model = "claude-3-opus-20240229"

    def generate_response(self, prompt: str) -> str:
        try:
            if self.provider == "openai":
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
            else:
                message = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.7,
                    system="You are a startup and industry analysis expert.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content

        except Exception as e:
            logger.error(f"AI generation failed: {str(e)}")
            raise

@st.cache_data(ttl=3600)
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
            results = list(ddgs.text(search_query, max_results=10))
            
            analysis_prompt = f"""Analyze these competitors in {industry}:
            {json.dumps(results)}
            
            Identify the top 3 most relevant direct competitors.
            Return a JSON array with exactly 3 companies, each containing:
            {{
                "name": "Company Name",
                "website": "company website",
                "description": "2-sentence description",
                "differentiator": "key unique selling point"
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
        st.error(f"Error finding competitors: {str(e)}")
        return []

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
     # Add style for the banner image
    st.markdown("""
        <style>
            img {
                border-radius: 20px;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Add banner image at the top using PIL
    try:
        response = requests.get("https://drive.google.com/uc?id=1Ed8JyPQzi-wkFu6KL6I7toAw4mQh064S", stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading banner image: {str(e)}")
    
    # Input section with improved layout
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            startup_name = st.text_input(
                "Startup Name",
                help="Enter your startup's name",
                placeholder="e.g., TechVenture Inc."
            )
        
        with col2:
            pitch = st.text_area(
                "One-Sentence Pitch",
                help="Describe what your startup does in one sentence",
                max_chars=200,
                placeholder="e.g., We provide AI-powered analytics for small businesses"
            )
    
        analyze_button = st.button("🔍 Analyze Market", use_container_width=True, type="primary")
    
    # Analysis section
    if analyze_button and startup_name and pitch:
        with st.status("Analyzing market...", expanded=True) as status:
            st.write("Identifying relevant industries...")
            st.session_state.industries = identify_industries(pitch)
            st.session_state.competitors = {}
            status.update(label="Analysis complete!", state="complete")
            st.rerun()

    # Results section
    if st.session_state.industries:
        st.divider()
        
        # Create tabs for industries
        tab_titles = st.session_state.industries
        tabs = st.tabs(tab_titles)
        
        # Handle tab content
        for i, tab in enumerate(tabs):
            with tab:
                industry = st.session_state.industries[i]
                
                # Load competitors if not already loaded
                if industry not in st.session_state.competitors:
                    with st.status(f"Analyzing competitors in {industry}...", expanded=True):
                        competitors = find_competitors(industry, pitch)
                        st.session_state.competitors[industry] = competitors
                
                # Display competitors using native components
                if industry in st.session_state.competitors:
                    for competitor in st.session_state.competitors[industry]:
                        render_competitor_card(competitor)
        
        # Export section
        if st.session_state.competitors:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                export_results(startup_name)

if __name__ == "__main__":
    main()
