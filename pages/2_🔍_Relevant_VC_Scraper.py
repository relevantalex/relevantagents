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

class SearchProvider:
    """Base class for search providers"""
    def search(self, query: str, max_results: int) -> List[Dict]:
        raise NotImplementedError

class DuckDuckGoProvider(SearchProvider):
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int) -> List[Dict]:
        try:
            results = self.ddgs.text(query, max_results=max_results)
            return [
                {
                    'title': r['title'],
                    'description': r['body'],
                    'url': r['link']
                } for r in results
            ]
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []

class BraveSearchProvider(SearchProvider):
    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY") or st.secrets.get("api_keys", {}).get("brave_api_key")
        if not self.api_key:
            logger.warning("Brave Search API key not found")
    
    def search(self, query: str, max_results: int) -> List[Dict]:
        if not self.api_key:
            return []
            
        try:
            headers = {"X-Subscription-Token": self.api_key}
            params = {
                "q": query,
                "count": min(max_results, 100)
            }
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        'title': result['title'],
                        'description': result.get('description', ''),
                        'url': result['url']
                    }
                    for result in data.get('web', {}).get('results', [])
                ]
            return []
        except Exception as e:
            logger.error(f"Brave search error: {str(e)}")
            return []

class GoogleSearchProvider(SearchProvider):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("api_keys", {}).get("google_api_key")
        self.cx = os.getenv("GOOGLE_CX") or st.secrets.get("api_keys", {}).get("google_cx")
        if not (self.api_key and self.cx):
            logger.warning("Google Search API credentials not found")
    
    def search(self, query: str, max_results: int) -> List[Dict]:
        if not (self.api_key and self.cx):
            return []
            
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            results = []
            
            # Google API returns max 10 results per request
            for start in range(1, min(max_results + 1, 101), 10):
                params = {
                    'key': self.api_key,
                    'cx': self.cx,
                    'q': query,
                    'start': start
                }
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        results.append({
                            'title': item['title'],
                            'description': item.get('snippet', ''),
                            'url': item['link']
                        })
                
                if len(results) >= max_results:
                    break
            
            return results[:max_results]
        except Exception as e:
            logger.error(f"Google search error: {str(e)}")
            return []

class CrunchbaseProvider:
    def __init__(self):
        self.api_key = os.getenv("CRUNCHBASE_API_KEY") or st.secrets.get("api_keys", {}).get("crunchbase_api_key")
        if not self.api_key:
            logger.warning("Crunchbase API key not found")
    
    def search(self, industry: str) -> List[Dict]:
        if not self.api_key:
            return []
            
        try:
            headers = {"X-cb-user-key": self.api_key}
            params = {
                "field_ids": ["name", "short_description", "website_url", "investor_type"],
                "query": [
                    {
                        "type": "predicate",
                        "field_id": "investor_type",
                        "operator_id": "includes",
                        "values": ["venture_capital"]
                    },
                    {
                        "type": "predicate",
                        "field_id": "investment_categories",
                        "operator_id": "includes",
                        "values": [industry]
                    }
                ]
            }
            
            response = requests.post(
                "https://api.crunchbase.com/api/v4/searches/organizations",
                headers=headers,
                json=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        'title': item['properties']['name'],
                        'description': item['properties'].get('short_description', ''),
                        'url': item['properties'].get('website_url', '')
                    }
                    for item in data.get('entities', [])
                ]
            return []
        except Exception as e:
            logger.error(f"Crunchbase search error: {str(e)}")
            return []

class VCSearchAgent:
    """Agent responsible for initial VC discovery using multiple sources"""
    def __init__(self):
        self.search_providers = [
            DuckDuckGoProvider(),
            BraveSearchProvider(),
            GoogleSearchProvider()
        ]
        self.crunchbase = CrunchbaseProvider()
        self.results = []
    
    async def search(self, industry: str, stage: str, max_results: int = 200) -> List[VCFirm]:
        all_results = []
        
        # 1. Search using multiple search engines
        search_terms = self._generate_search_terms(industry, stage)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # Submit search tasks for each provider and search term
            for provider in self.search_providers:
                for term in search_terms:
                    futures.append(
                        executor.submit(
                            provider.search,
                            term,
                            max_results // (len(self.search_providers) * len(search_terms))
                        )
                    )
            
            # Add Crunchbase search
            futures.append(executor.submit(self.crunchbase.search, industry))
            
            with st.spinner(f"🔍 Searching for VCs... (0/{len(futures)} queries complete)"):
                for i, future in enumerate(futures):
                    try:
                        results = future.result()
                        for result in results:
                            if self._is_potential_vc(result['title'], result['description']):
                                vc = VCFirm(
                                    name=result['title'],
                                    url=result['url'],
                                    description=result['description'],
                                    source='Search Results'
                                )
                                all_results.append(vc)
                        st.spinner(f"🔍 Searching for VCs... ({i+1}/{len(futures)} queries complete)")
                    except Exception as e:
                        logger.error(f"Error in search: {str(e)}")
        
        # Add specialized VC database searches
        all_results.extend(await self._search_unicorn_nest(industry))
        
        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for vc in all_results:
            domain = urlparse(vc.url).netloc
            if domain not in seen_urls:
                seen_urls.add(domain)
                unique_results.append(vc)
        
        st.success(f"Found {len(unique_results)} potential VCs across all sources")
        return unique_results

    async def _search_unicorn_nest(self, industry: str) -> List[VCFirm]:
        """Search VCs on Unicorn Nest"""
        results = []
        try:
            base_url = "https://unicorn-nest.com/funds/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try multiple search variations
            search_terms = [industry] + industry.split()
            for term in search_terms:
                try:
                    response = requests.get(
                        f"{base_url}?search={term}",
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for fund in soup.find_all(class_='fund-card'):
                            try:
                                vc = VCFirm(
                                    name=fund.find('h3').text.strip(),
                                    url=urljoin(base_url, fund.find('a')['href']),
                                    description=fund.find(class_='description').text.strip(),
                                    source='Unicorn Nest'
                                )
                                results.append(vc)
                            except Exception:
                                continue
                except Exception as e:
                    logger.error(f"Error searching Unicorn Nest term {term}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error accessing Unicorn Nest: {str(e)}")
        
        return results

    def _generate_search_terms(self, industry: str, stage: str) -> List[str]:
        """Generate comprehensive search terms"""
        terms = [
            f"{industry} venture capital",
            f"{industry} VC firms",
            f"{industry} investors",
            f"top {industry} VCs",
            f"{stage} stage {industry} investors",
            f"{industry} focused venture capital",
            "venture capital firms directory",
            "VC firms database",
            "venture capital investors list",
            "active venture capital firms",
            f"who invests in {industry}",
            f"leading {industry} investors",
            f"{industry} investment firms",
            "technology venture capital"
        ]
        
        # Add variations
        industry_terms = industry.split()
        for term in industry_terms:
            terms.extend([
                f"{term} investors",
                f"{term} venture capital",
                f"VCs investing in {term}",
                f"{term} focused funds"
            ])
        
        return list(set(terms))

    def _is_potential_vc(self, title: str, description: str) -> bool:
        """Improved check for potential VC firms"""
        text = (title + ' ' + description).lower()
        
        # Skip obvious non-VCs
        skip_terms = [
            'wikipedia', 'dictionary', '.gov', 'news article',
            'press release', 'job posting', 'linkedin.com/jobs'
        ]
        if any(term in text for term in skip_terms):
            return False
        
        # Check for VC-related terms
        vc_terms = [
            'venture', 'capital', 'vc', 'investor', 'investment',
            'fund', 'equity', 'portfolio', 'startup', 'ventures',
            'partners', 'investments', 'venture partners',
            'capital partners', 'investment firm'
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
