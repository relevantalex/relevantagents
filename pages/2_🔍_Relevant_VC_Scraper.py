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
import re
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VCScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.ddgs = DDGS()
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def find_relevant_vcs(self, startup_data: Dict[str, str]) -> List[Dict]:
        """Find VCs that match the startup's profile with broader industry matching"""
        industry = startup_data['industry']
        stage = startup_data['stage']
        
        # Get broader industry terms using GPT-4
        broader_terms = self._get_broader_industry_terms(industry)
        
        # Generate search queries with broader scope
        queries = []
        
        # Add direct industry queries
        queries.extend([
            f"{industry} investors",
            f"{industry} venture capital",
            f"VCs investing {industry}",
            f"top {industry} venture capital firms"
        ])
        
        # Add broader industry queries
        for term in broader_terms:
            queries.extend([
                f"{term} investors",
                f"{term} venture capital",
                f"VCs investing {term}"
            ])
        
        # Add stage-specific queries if stage is provided
        if stage and stage.lower() not in ['not specified', 'unknown']:
            queries.extend([
                f"{industry} {stage} investors",
                f"{stage} stage venture capital firms"
            ])
        
        results = []
        for query in queries:
            try:
                # Increase max_results for broader coverage
                search_results = self.ddgs.text(query, max_results=10)
                for result in search_results:
                    if self._is_vc_result(result['title'], result['body']):
                        results.append({
                            'name': result['title'],
                            'description': result['body'],
                            'url': result['link']
                        })
            except Exception as e:
                logger.error(f"Error searching for VCs: {str(e)}")
                continue
        
        return self._deduplicate_results(results)

    def _get_broader_industry_terms(self, industry: str) -> List[str]:
        """Get broader and related industry terms using GPT-4"""
        prompt = f"""
        For the industry "{industry}", provide a JSON array of broader and related industry terms.
        Include both broader categories and related sectors. Return only the JSON array, nothing else.
        Example: if industry is "AI chatbots", return ["artificial intelligence", "machine learning", "conversational AI", "enterprise software", "natural language processing", "SaaS"]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding industry categories and related sectors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('terms', [industry])
        except Exception as e:
            logger.error(f"Error getting broader industry terms: {str(e)}")
            return [industry]

    def _is_vc_result(self, title: str, body: str) -> bool:
        """Check if search result is likely a VC firm with broader matching"""
        vc_terms = [
            'venture', 'capital', 'vc', 'investor', 'investment',
            'fund', 'equity', 'portfolio', 'startup', 'ventures',
            'partners', 'capital partners', 'investments'
        ]
        text = (title + ' ' + body).lower()
        return any(term in text for term in vc_terms)

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate VC firms"""
        seen = set()
        unique_results = []
        for result in results:
            domain = urlparse(result['url']).netloc
            if domain not in seen:
                seen.add(domain)
                unique_results.append(result)
        return unique_results

    def scrape_vc_info(self, vc: Dict) -> Dict:
        """Scrape detailed information about a VC firm"""
        try:
            response = requests.get(vc['url'], headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract contact information
            emails = self._extract_emails(response.text)
            linkedin = self._extract_linkedin_profiles(soup)
            
            # Extract investment focus
            focus = self._extract_investment_focus(soup)
            
            return {
                **vc,
                'emails': emails,
                'linkedin_profiles': linkedin,
                'investment_focus': focus
            }
        except Exception as e:
            logger.error(f"Error scraping VC info: {str(e)}")
            return vc

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))

    def _extract_linkedin_profiles(self, soup) -> List[str]:
        """Extract LinkedIn profile URLs"""
        linkedin_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'linkedin.com/in/' in href:
                linkedin_links.append(href)
        return list(set(linkedin_links))

    def _extract_investment_focus(self, soup) -> str:
        """Extract investment focus information"""
        # Look for common sections that might contain investment focus
        focus_keywords = ['investment', 'focus', 'strategy', 'thesis']
        text_blocks = []
        
        for keyword in focus_keywords:
            elements = soup.find_all(['p', 'div', 'section'], 
                                  string=re.compile(keyword, re.I))
            for element in elements:
                text_blocks.append(element.get_text().strip())
        
        return ' '.join(text_blocks)[:500] if text_blocks else ""

    def generate_outreach_email(self, vc_info: Dict, startup_data: Dict) -> str:
        """Generate a personalized outreach email"""
        prompt = f"""
        Create a concise, personalized email to a VC.
        
        VC Firm: {vc_info['name']}
        VC Focus: {vc_info.get('investment_focus', 'Not available')}
        
        Startup:
        Name: {startup_data['name']}
        Industry: {startup_data['industry']}
        Stage: {startup_data['stage']}
        Pitch: {startup_data['pitch']}
        
        Requirements:
        1. Keep it under 200 words
        2. Mention specific alignment with VC's focus
        3. Include key metrics or achievements
        4. End with a clear call to action
        5. Mention that a one-pager is attached
        
        Return only the email text, no subject line.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at writing concise, effective VC outreach emails."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}")
            return ""

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
    st.caption("Find and reach out to the perfect VCs for your startup")
    
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
    
    # Initialize scraper
    scraper = VCScraper()
    
    # Start search button
    if st.button("🔎 Find Matching VCs", use_container_width=True):
        with st.spinner("Searching for relevant VCs..."):
            # Find relevant VCs
            vcs = scraper.find_relevant_vcs(startup_data)
            
            if not vcs:
                st.warning("No matching VCs found. Try adjusting your startup profile.")
                return
            
            # Process each VC
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, vc in enumerate(vcs):
                progress = (idx + 1) / len(vcs)
                progress_bar.progress(progress)
                status_text.text(f"Processing {vc['name']}...")
                
                # Scrape detailed info
                vc_info = scraper.scrape_vc_info(vc)
                
                # Generate outreach email
                outreach_email = scraper.generate_outreach_email(vc_info, startup_data)
                
                results.append({
                    **vc_info,
                    'outreach_email': outreach_email
                })
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.subheader("📊 Results")
            
            for vc in results:
                with st.expander(f"🏢 {vc['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Website:**", vc['url'])
                        st.write("**Description:**", vc['description'])
                        if vc.get('investment_focus'):
                            st.write("**Investment Focus:**", vc['investment_focus'])
                    
                    with col2:
                        if vc.get('emails'):
                            st.write("**Contact Emails:**")
                            for email in vc['emails']:
                                st.write(f"- {email}")
                        
                        if vc.get('linkedin_profiles'):
                            st.write("**LinkedIn Profiles:**")
                            for profile in vc['linkedin_profiles']:
                                st.write(f"- {profile}")
                    
                    st.write("**📧 Suggested Outreach Email:**")
                    st.text_area("Email Text", vc['outreach_email'], height=200, key=f"email_{vc['name']}")
            
            # Create downloadable results
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                "vc_outreach_list.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()
