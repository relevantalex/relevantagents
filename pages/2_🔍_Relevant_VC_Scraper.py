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
from duckduckgo_search import DDGS
import openai
from anthropic import Anthropic

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VCResearchEngine:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.ddgs = DDGS()

    def search_vcs(self, startup_info: Dict[str, any]) -> List[Dict[str, any]]:
        """Search for relevant VCs based on startup information"""
        industry = startup_info.get('industry', '')
        stage = startup_info.get('stage', '')
        location = startup_info.get('location', '')
        
        search_queries = [
            f"venture capital firms investing in {industry} startups",
            f"VC funds {stage} stage {industry}",
            f"top venture capital firms {location} {industry}",
            f"early stage investors {industry} technology",
        ]
        
        results = []
        for query in search_queries:
            try:
                ddg_results = self.ddgs.text(query, max_results=5)
                for result in ddg_results:
                    if 'venture' in result['title'].lower() or 'capital' in result['title'].lower():
                        results.append({
                            'name': result['title'],
                            'website': result['link'],
                            'description': result['body']
                        })
            except Exception as e:
                logger.error(f"Error in DuckDuckGo search: {str(e)}")
                
        return self.deduplicate_results(results)
    
    def deduplicate_results(self, results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Remove duplicate VCs based on website domain"""
        seen_domains = set()
        unique_results = []
        
        for result in results:
            domain = urlparse(result['website']).netloc
            if domain not in seen_domains:
                seen_domains.add(domain)
                unique_results.append(result)
        
        return unique_results

    def enrich_vc_data(self, vc_data: Dict[str, any]) -> Dict[str, any]:
        """Enrich VC data with additional information from various sources"""
        website = vc_data.get('website', '')
        name = vc_data.get('name', '')
        
        # Search for additional information
        search_queries = [
            f"{name} venture capital portfolio companies",
            f"{name} VC investment thesis",
            f"{name} VC team members linkedin",
            f"{name} recent investments news"
        ]
        
        additional_info = {
            'portfolio_companies': [],
            'investment_thesis': '',
            'team_members': [],
            'recent_investments': []
        }
        
        for query in search_queries:
            try:
                results = self.ddgs.text(query, max_results=3)
                for result in results:
                    if name.lower() in result['title'].lower():
                        if 'portfolio' in query:
                            additional_info['portfolio_companies'].append(result['body'])
                        elif 'thesis' in query:
                            additional_info['investment_thesis'] += result['body']
                        elif 'team' in query:
                            additional_info['team_members'].append(result['body'])
                        elif 'investments' in query:
                            additional_info['recent_investments'].append(result['body'])
            except Exception as e:
                logger.error(f"Error enriching VC data for {name}: {str(e)}")
        
        return additional_info

class VCAnalyzer:
    def __init__(self):
        self.ai_provider = "openai"  # Default to OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")
        self.model = "gpt-4-turbo-preview"

    def analyze_vc_fit(self, website_data: Dict[str, str], startup_pitch: str) -> Dict[str, any]:
        """Analyze how well a VC fits with a startup"""
        try:
            # Prepare the prompt
            prompt = f"""
            Analyze how well this VC firm might fit with the following startup:

            Startup Pitch: {startup_pitch}

            VC Information:
            {website_data}

            Please analyze:
            1. Investment Focus Alignment
            2. Stage Alignment
            3. Potential Concerns
            4. Overall Fit Score (0-100)
            """

            # Get completion from OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a VC analysis expert helping evaluate potential investor fits."},
                    {"role": "user", "content": prompt}
                ]
            )

            analysis = response.choices[0].message.content

            # Extract the fit score using regex
            score_match = re.search(r'Overall Fit Score.*?(\d+)', analysis)
            fit_score = int(score_match.group(1)) if score_match else 0

            return {
                'analysis': analysis,
                'fit_score': fit_score
            }

        except Exception as e:
            logger.error(f"Error in VC analysis: {str(e)}")
            return {
                'analysis': f"Error performing analysis: {str(e)}",
                'fit_score': 0
            }

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

def find_relevant_vcs(startup_data: Dict[str, any]):
    """Find relevant VCs through internet research"""
    vc_researcher = VCResearchEngine()
    vc_analyzer = VCAnalyzer()
    
    with st.spinner("Searching for relevant VCs..."):
        # Search for VCs
        results = vc_researcher.search_vcs(startup_data)
        
        if not results:
            st.warning("No VCs found matching your criteria.")
            return
        
        # Process results
        processed_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, vc in enumerate(results):
            try:
                progress = (idx + 1) / len(results)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {vc['name']}...")
                
                # Enrich VC data
                additional_info = vc_researcher.enrich_vc_data(vc)
                
                # Analyze fit
                analysis_result = vc_analyzer.analyze_vc_fit(
                    {**vc, **additional_info},
                    startup_data['pitch']
                )
                
                processed_results.append({
                    **vc,
                    **additional_info,
                    'analysis': analysis_result['analysis'],
                    'fit_score': analysis_result['fit_score']
                })
                
            except Exception as e:
                logger.error(f"Error processing VC {vc['name']}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort results by fit score
        processed_results.sort(key=lambda x: x['fit_score'], reverse=True)
        
        # Display results
        st.subheader("Found VCs")
        
        for vc in processed_results:
            with st.expander(f"🏢 {vc['name']} (Fit Score: {vc['fit_score']})"):
                st.write("**Website:**", vc['website'])
                st.write("**Description:**", vc['description'])
                
                # Show portfolio companies if available
                if vc['portfolio_companies']:
                    st.write("**Notable Portfolio Companies:**")
                    for company in vc['portfolio_companies'][:3]:  # Show top 3
                        st.write(f"- {company}")
                
                # Show investment thesis if available
                if vc['investment_thesis']:
                    st.write("**Investment Thesis:**", vc['investment_thesis'][:500] + "...")
                
                # Show analysis
                st.write("**Analysis:**", vc['analysis'])
        
        # Create downloadable results
        results_df = pd.DataFrame(processed_results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results as CSV",
            csv,
            "vc_research_results.csv",
            "text/csv",
            key='download-csv'
        )

def main():
    st.set_page_config(
        page_title="Relevant VC Scraper",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Relevant VC Scraper")
    st.caption("Find and analyze VCs that match your startup's profile")
    
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
    
    # Start search button
    if st.button("🔎 Start VC Search", use_container_width=True):
        find_relevant_vcs(startup_data)

if __name__ == "__main__":
    main()
