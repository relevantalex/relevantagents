import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import os
from database import DatabaseManager
import time
import json
from duckduckgo_search import DDGS
import openai
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VCFirm:
    name: str
    url: str
    description: str = ""
    investment_focus: str = ""
    emails: List[str] = None
    linkedin_profiles: List[str] = None
    relevance_score: float = 0.0
    source: str = ""
    
    def __post_init__(self):
        self.emails = self.emails or []
        self.linkedin_profiles = self.linkedin_profiles or []

class VCSearchAgent:
    """Agent responsible for initial VC discovery"""
    def __init__(self):
        self.ddgs = DDGS()
        self.results = []
        
    async def search(self, industry: str, stage: str, max_results: int = 200) -> List[VCFirm]:
        search_terms = self._generate_search_terms(industry, stage)
        all_results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for term in search_terms:
                futures.append(
                    executor.submit(self._search_term, term, max_results // len(search_terms))
                )
            
            with st.spinner(f"🔍 Searching for VCs... (0/{len(futures)} queries complete)"):
                for i, future in enumerate(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                        st.spinner(f"🔍 Searching for VCs... ({i+1}/{len(futures)} queries complete)")
                    except Exception as e:
                        logger.error(f"Error in search: {str(e)}")
        
        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for vc in all_results:
            if vc.url not in seen_urls:
                seen_urls.add(vc.url)
                unique_results.append(vc)
        
        return unique_results

    def _generate_search_terms(self, industry: str, stage: str) -> List[str]:
        """Generate comprehensive search terms"""
        terms = [
            f"{industry} venture capital",
            f"{industry} VC firms",
            f"{industry} investors",
            f"top {industry} VCs",
            f"{stage} stage {industry} investors",
            f"{industry} focused venture capital",
            "technology venture capital firms",
            "startup investors directory",
            "venture capital directory",
            "VC firms list"
        ]
        
        # Add variations
        industry_terms = industry.split()
        for term in industry_terms:
            terms.extend([
                f"{term} investors",
                f"{term} venture capital",
                f"VCs investing in {term}"
            ])
        
        return list(set(terms))

    def _search_term(self, term: str, max_results: int) -> List[VCFirm]:
        """Search for a single term"""
        results = []
        try:
            search_results = self.ddgs.text(term, max_results=max_results)
            for result in search_results:
                if self._is_potential_vc(result['title'], result['body']):
                    vc = VCFirm(
                        name=result['title'],
                        url=result['link'],
                        description=result['body'],
                        source='Web Search'
                    )
                    results.append(vc)
        except Exception as e:
            logger.error(f"Error searching term '{term}': {str(e)}")
        
        return results

    def _is_potential_vc(self, title: str, body: str) -> bool:
        """Loose check for potential VC firms"""
        text = (title + ' ' + body).lower()
        
        # Skip obvious non-VCs
        skip_terms = ['wikipedia', 'dictionary', '.gov', 'news article']
        if any(term in text for term in skip_terms):
            return False
        
        # Check for VC-related terms
        vc_terms = [
            'venture', 'capital', 'vc', 'investor', 'investment',
            'fund', 'equity', 'portfolio', 'startup', 'ventures'
        ]
        return any(term in text for term in vc_terms)

class VCEnrichmentAgent:
    """Agent responsible for enriching VC data with website information"""
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        )
    
    async def enrich_vcs(self, vcs: List[VCFirm], industry: str) -> List[VCFirm]:
        """Enrich VC firms with website data and relevance scores"""
        enriched_vcs = []
        total = len(vcs)
        
        with st.spinner("🔍 Analyzing VC websites..."):
            progress_bar = st.progress(0)
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for vc in vcs:
                    futures.append(
                        executor.submit(self._enrich_vc, vc, industry)
                    )
                
                for i, future in enumerate(futures):
                    try:
                        enriched_vc = future.result()
                        if enriched_vc:
                            enriched_vcs.append(enriched_vc)
                        progress_bar.progress((i + 1) / total)
                    except Exception as e:
                        logger.error(f"Error enriching VC {vcs[i].name}: {str(e)}")
        
        return sorted(enriched_vcs, key=lambda x: x.relevance_score, reverse=True)

    def _enrich_vc(self, vc: VCFirm, industry: str) -> Optional[VCFirm]:
        """Enrich a single VC firm"""
        try:
            # Scrape website content
            content = self._scrape_website(vc.url)
            if not content:
                return None
            
            # Extract investment focus
            vc.investment_focus = self._extract_investment_focus(content)
            
            # Calculate relevance score
            vc.relevance_score = self._calculate_relevance(
                vc.description + " " + vc.investment_focus,
                industry
            )
            
            return vc
        except Exception as e:
            logger.error(f"Error enriching {vc.name}: {str(e)}")
            return None

    def _scrape_website(self, url: str) -> Optional[str]:
        """Scrape website content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def _extract_investment_focus(self, content: str) -> str:
        """Extract investment focus from website content"""
        try:
            prompt = f"""
            Extract the investment focus from this VC firm's website content.
            Focus on: industries, stages, check sizes, and geographic preferences.
            
            Content: {content[:2000]}  # Limit content length
            
            Return a concise summary (max 200 words).
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing VC firm websites."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error extracting investment focus: {str(e)}")
            return ""

    def _calculate_relevance(self, vc_text: str, industry: str) -> float:
        """Calculate relevance score"""
        try:
            prompt = f"""
            Rate how relevant this VC firm is for a startup in the {industry} industry.
            
            VC Description: {vc_text[:1000]}
            
            Return only a number between 0 and 1, where 1 is highly relevant.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at matching VCs to startups."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0), 1)  # Ensure between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 0.0

class VCContactAgent:
    """Agent responsible for finding contact information"""
    def __init__(self):
        pass
    
    async def find_contacts(self, vcs: List[VCFirm]) -> List[VCFirm]:
        """Find contact information for VC firms"""
        with st.spinner("📧 Finding contact information..."):
            progress_bar = st.progress(0)
            total = len(vcs)
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for vc in vcs:
                    futures.append(
                        executor.submit(self._find_vc_contacts, vc)
                    )
                
                for i, future in enumerate(futures):
                    try:
                        vc = future.result()
                        progress_bar.progress((i + 1) / total)
                    except Exception as e:
                        logger.error(f"Error finding contacts: {str(e)}")
        
        return vcs

    def _find_vc_contacts(self, vc: VCFirm) -> VCFirm:
        """Find contacts for a single VC firm"""
        try:
            # Get all pages to search
            pages_to_search = self._get_contact_pages(vc.url)
            
            for url in pages_to_search:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Extract emails
                        emails = self._extract_emails(response.text)
                        vc.emails.extend(emails)
                        
                        # Extract LinkedIn profiles
                        soup = BeautifulSoup(response.text, 'html.parser')
                        linkedin_profiles = self._extract_linkedin_profiles(soup)
                        vc.linkedin_profiles.extend(linkedin_profiles)
                except Exception as e:
                    logger.error(f"Error processing page {url}: {str(e)}")
                    continue
            
            # Remove duplicates
            vc.emails = list(set(vc.emails))
            vc.linkedin_profiles = list(set(vc.linkedin_profiles))
            
            return vc
        except Exception as e:
            logger.error(f"Error finding contacts for {vc.name}: {str(e)}")
            return vc

    def _get_contact_pages(self, base_url: str) -> List[str]:
        """Get URLs of pages likely to contain contact information"""
        contact_paths = [
            '/contact', '/contact-us', '/team', '/about', '/about-us',
            '/people', '/our-team', '/partners', '/investors'
        ]
        
        pages = [base_url]  # Always include homepage
        parsed_url = urlparse(base_url)
        base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        for path in contact_paths:
            pages.append(urljoin(base, path))
        
        return pages

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
        return linkedin_links

async def find_relevant_vcs(startup_data: Dict[str, str]) -> List[VCFirm]:
    """Main function to find and analyze VC firms"""
    # Initialize agents
    search_agent = VCSearchAgent()
    enrichment_agent = VCEnrichmentAgent()
    contact_agent = VCContactAgent()
    
    # Phase 1: Initial VC Discovery
    st.subheader("Phase 1: Initial VC Discovery")
    vcs = await search_agent.search(
        industry=startup_data['industry'],
        stage=startup_data['stage']
    )
    st.info(f"Found {len(vcs)} potential VC firms")
    
    # Phase 2: Enrich VC Data
    st.subheader("Phase 2: Analyzing VC Firms")
    vcs = await enrichment_agent.enrich_vcs(vcs, startup_data['industry'])
    st.info(f"Analyzed {len(vcs)} VC firms")
    
    # Phase 3: Find Contact Information
    st.subheader("Phase 3: Finding Contact Information")
    vcs = await contact_agent.find_contacts(vcs)
    
    return vcs

def display_results(vcs: List[VCFirm]):
    """Display results in a nice format"""
    st.header("🎯 Results")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total VCs Found", len(vcs))
    with col2:
        relevant_vcs = len([vc for vc in vcs if vc.relevance_score > 0.7])
        st.metric("Highly Relevant VCs", relevant_vcs)
    with col3:
        vcs_with_contacts = len([vc for vc in vcs if vc.emails or vc.linkedin_profiles])
        st.metric("VCs with Contacts", vcs_with_contacts)
    
    # Display VCs in expandable cards
    for vc in vcs:
        with st.expander(f"🏢 {vc.name} (Relevance: {vc.relevance_score:.2f})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Website:**", vc.url)
                st.write("**Description:**", vc.description)
                if vc.investment_focus:
                    st.write("**Investment Focus:**", vc.investment_focus)
            
            with col2:
                if vc.emails:
                    st.write("**📧 Contact Emails:**")
                    for email in vc.emails:
                        st.write(f"- {email}")
                
                if vc.linkedin_profiles:
                    st.write("**👥 LinkedIn Profiles:**")
                    for profile in vc.linkedin_profiles:
                        st.write(f"- {profile}")
    
    # Export results
    if vcs:
        df = pd.DataFrame([vars(vc) for vc in vcs])
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "vc_outreach_list.csv",
            "text/csv",
            key='download-csv'
        )

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
    
    # Start search button
    if st.button("🔎 Find Matching VCs", use_container_width=True):
        vcs = asyncio.run(find_relevant_vcs(startup_data))
        display_results(vcs)

if __name__ == "__main__":
    main()
