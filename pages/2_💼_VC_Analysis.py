import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
from io import StringIO
import csv
from urllib.parse import urlparse
import re
from database import DatabaseManager
import time
from anthropic import Anthropic
import openai
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VCData:
    def __init__(self):
        self.name: str = ""
        self.website: str = ""
        self.email: str = ""
        self.summary: str = ""
        self.match_reason: str = ""
        self.partner_linkedin: str = ""
        self.analyst_linkedin: str = ""
        self.portfolio_companies: List[str] = []
        self.investment_verticals: List[str] = []
        self.match_score: float = 0.0

class VCScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_emails(self, text: str) -> List[str]:
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))

    def scrape_website(self, url: str) -> Dict[str, any]:
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            
            emails = self.extract_emails(text)
            
            return {
                'text': text[:5000],  # Limit text length
                'emails': emails
            }
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {'text': '', 'emails': []}

class VCAnalyzer:
    def __init__(self):
        self.ai_provider = "openai"  # Default to OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        self.model = "gpt-4-turbo-preview"

    def analyze_vc_fit(self, vc_data: Dict[str, str], startup_pitch: str) -> Dict[str, any]:
        prompt = f"""
        Analyze this VC firm based on the following information and determine if it's a good fit for our startup.
        
        VC Information:
        {vc_data.get('text', '')}
        
        Startup Pitch:
        {startup_pitch}
        
        Please provide:
        1. A summary of the VC's focus areas and investment strategy
        2. A list of relevant portfolio companies
        3. Key investment verticals
        4. A match score (0-100)
        5. If score > 70, explain why this fund is a good match
        
        Format the response as JSON with keys: summary, portfolio_companies, verticals, match_score, match_reason
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return eval(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                'summary': '',
                'portfolio_companies': [],
                'verticals': [],
                'match_score': 0,
                'match_reason': ''
            }

def process_vc_list(uploaded_file, startup_pitch: str) -> pd.DataFrame:
    vc_scraper = VCScraper()
    vc_analyzer = VCAnalyzer()
    results = []
    
    df = pd.read_csv(uploaded_file)
    total_vcs = len(df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        progress = (idx + 1) / total_vcs
        progress_bar.progress(progress)
        status_text.text(f"Processing VC {idx + 1} of {total_vcs}: {row.get('name', 'Unknown')}")
        
        vc_data = VCData()
        vc_data.name = row.get('name', '')
        vc_data.website = row.get('website', '')
        
        # Scrape website
        website_data = vc_scraper.scrape_website(vc_data.website)
        
        # Analyze VC fit
        analysis = vc_analyzer.analyze_vc_fit(website_data, startup_pitch)
        
        if analysis['match_score'] >= 70:  # Only include high-matching VCs
            results.append({
                'VC Firm Name': vc_data.name,
                'Contact Email': ', '.join(website_data['emails']),
                'Why This Fund is a Match': analysis['match_reason'],
                'Partner LinkedIn': '',  # To be filled manually or through LinkedIn API
                'Analyst LinkedIn': ''   # To be filled manually or through LinkedIn API
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    st.title("🎯 VC Analysis")
    
    if 'startup_pitch' not in st.session_state:
        st.session_state.startup_pitch = ""
    
    with st.expander("📝 Enter Your Startup Pitch", expanded=not bool(st.session_state.startup_pitch)):
        startup_pitch = st.text_area(
            "Describe your startup, target market, and unique value proposition",
            value=st.session_state.startup_pitch,
            height=200
        )
        if startup_pitch != st.session_state.startup_pitch:
            st.session_state.startup_pitch = startup_pitch
    
    uploaded_file = st.file_uploader(
        "Upload your VC list (CSV with columns: name, website)",
        type=['csv']
    )
    
    if uploaded_file and st.session_state.startup_pitch:
        if st.button("Start VC Analysis"):
            with st.spinner("Analyzing VCs..."):
                results_df = process_vc_list(uploaded_file, st.session_state.startup_pitch)
                
                if not results_df.empty:
                    st.success(f"Found {len(results_df)} matching VCs!")
                    st.dataframe(results_df)
                    
                    # Export results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "vc_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No matching VCs found. Try adjusting your startup pitch or uploading a different VC list.")
    
    else:
        st.info("Please upload a CSV file with VC information and provide your startup pitch to begin the analysis.")

if __name__ == "__main__":
    main()
