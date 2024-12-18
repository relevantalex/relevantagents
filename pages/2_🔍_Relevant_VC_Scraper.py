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
        
        results = []
        
        # 1. Search Unicorn Nest
        st.info("Searching Unicorn Nest database...")
        unicorn_results = self._search_unicorn_nest(industry)
        results.extend(unicorn_results)
        
        # 2. Web Search with very loose filters
        st.info("Performing web search...")
        web_results = self._broad_web_search(industry, stage)
        results.extend(web_results)
        
        # Deduplicate results
        unique_results = self._deduplicate_results(results)
        
        st.info(f"Found {len(unique_results)} potential VCs")
        return unique_results

    def _search_unicorn_nest(self, industry: str) -> List[Dict]:
        """Search VCs on Unicorn Nest"""
        try:
            # Base URL for Unicorn Nest funds
            base_url = "https://unicorn-nest.com/funds/"
            
            # Make request to get the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            # Try different search variations
            results = []
            search_terms = [industry] + industry.split()  # Split industry into individual words
            
            for term in search_terms:
                try:
                    search_url = f"{base_url}?search={term}"
                    response = requests.get(search_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find fund entries
                        fund_elements = soup.find_all('div', class_='fund-card')  # Adjust class based on actual HTML
                        
                        for fund in fund_elements:
                            try:
                                name = fund.find('h3').text.strip()
                                description = fund.find('p', class_='description').text.strip()
                                url = "https://unicorn-nest.com" + fund.find('a')['href']
                                
                                results.append({
                                    'name': name,
                                    'description': description,
                                    'url': url,
                                    'source': 'Unicorn Nest'
                                })
                            except Exception as e:
                                continue
                except Exception as e:
                    logger.error(f"Error searching Unicorn Nest term {term}: {str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"Error accessing Unicorn Nest: {str(e)}")
            return []

    def _broad_web_search(self, industry: str, stage: str) -> List[Dict]:
        """Perform a broad web search for VCs with very loose filters"""
        try:
            # Get broader industry terms
            broader_terms = self._get_broader_industry_terms(industry)
            
            # Generate very broad queries
            queries = []
            
            # Industry-based queries
            base_terms = [industry] + broader_terms
            for term in base_terms:
                queries.extend([
                    f"{term} investors",
                    f"{term} VC",
                    f"venture capital {term}",
                    f"investment firms {term}",
                    "top venture capital firms",
                    "active venture capital investors",
                    "technology investors",
                    "startup investors"
                ])
            
            results = []
            for query in queries:
                try:
                    # Increase max_results significantly
                    search_results = self.ddgs.text(query, max_results=20)
                    
                    if not search_results:
                        st.warning(f"No results found for query: {query}")
                        continue
                        
                    for result in search_results:
                        # Very loose filtering - accept almost anything that might be a VC
                        if not self._is_obviously_not_vc(result['title'], result['body']):
                            results.append({
                                'name': result['title'],
                                'description': result['body'],
                                'url': result['link'],
                                'source': 'Web Search'
                            })
                except Exception as e:
                    logger.error(f"Error in web search for query '{query}': {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in broad web search: {str(e)}")
            return []

    def _is_obviously_not_vc(self, title: str, body: str) -> bool:
        """Very loose check - only filter out obvious non-VCs"""
        # Convert to lower case for comparison
        text = (title + ' ' + body).lower()
        
        # List of terms that indicate this is definitely not a VC
        obvious_non_vc = [
            'wikipedia',
            'dictionary',
            'definition',
            '.gov',
            'news article',
            'press release'
        ]
        
        return any(term in text for term in obvious_non_vc)

    def _get_broader_industry_terms(self, industry: str) -> List[str]:
        """Get much broader industry terms"""
        prompt = f"""
        For the industry "{industry}", provide a JSON object with an array of related terms.
        Include:
        1. The industry itself
        2. Broader categories
        3. Related sectors
        4. Technology areas
        5. Market segments
        6. Generic terms like "technology", "software", "digital"
        
        Format: {{"terms": ["term1", "term2", ...]}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding industry categories and related sectors. Be comprehensive and include broader terms."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            terms = result.get('terms', [industry])
            
            # Add some generic tech investment terms
            terms.extend([
                "technology",
                "software",
                "digital",
                "innovation",
                "startup",
                "tech company"
            ])
            
            return list(set(terms))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error getting broader industry terms: {str(e)}")
            return [industry, "technology", "software", "startup"]

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
