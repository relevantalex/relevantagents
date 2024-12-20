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
import aiohttp
import ssl
import certifi

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
    stage_preference: str = ""
    
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
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 2))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                    
                if response.status_code != 200:
                    logger.error(f"Google Search API error: {response.status}")
                    break
                    
                data = response.json()
                if 'items' not in data:
                    break
                    
                results.extend(data['items'])
                if len(data['items']) < 10:  # No more results
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
    """Agent responsible for discovering VCs using DuckDuckGo and GPT-4 validation"""
    def __init__(self):
        self.results = []
        self.cache = {}
        self.ddgs = DDGS()
        
        # Search templates for different platforms
        self.search_templates = {
            'linkedin_company': 'site:linkedin.com/company "venture capital" {industry} {stage}',
            'linkedin_people': 'site:linkedin.com/in "venture capital" "partner" {industry} {stage}',
            'wellfound': 'site:wellfound.com "venture capital" {industry} {stage}',
            'crunchbase': 'site:crunchbase.com/organization "venture capital" {industry} {stage}',
            'pitchbook': 'site:pitchbook.com/profiles "venture capital" {industry} {stage}'
        }

    async def search(self, industry: str, stage: str, max_results: int = 200):
        """Execute multi-platform search and validate results with GPT-4"""
        st.write("🔍 Searching for VC firms...")
        progress_bar = st.progress(0)
        all_results = []
        
        # Calculate results per template
        results_per_template = max_results // len(self.search_templates)
        
        for idx, (platform, template) in enumerate(self.search_templates.items(), 1):
            try:
                query = template.format(industry=industry, stage=stage)
                cache_key = f"{query}_{results_per_template}"
                
                if cache_key in self.cache:
                    results = self.cache[cache_key]
                    st.write(f"📎 Using cached results for {platform}")
                else:
                    st.write(f"🔎 Searching {platform}...")
                    results = await self._duckduckgo_search(query, results_per_template)
                    self.cache[cache_key] = results
                
                if results:
                    all_results.extend(results)
                    st.write(f"✅ Found {len(results)} results from {platform}")
                else:
                    st.write(f"⚠️ No results from {platform}")
                
                progress_bar.progress(idx / len(self.search_templates))
                
            except Exception as e:
                logger.error(f"Error searching {platform}: {str(e)}")
                st.write(f"❌ Error searching {platform}")
                continue
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        st.write(f"📊 Found {len(unique_results)} unique VC firms")
        
        # Process results in batches
        st.write("🔄 Enriching VC firm data...")
        enriched_results = []
        batch_size = 10
        total_batches = (len(unique_results) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(unique_results), batch_size), 1):
            batch = unique_results[i:i + batch_size]
            enrichment_tasks = []
            
            for result in batch:
                task = asyncio.create_task(self._validate_and_enrich(result, industry, stage))
                enrichment_tasks.append(task)
            
            batch_results = await asyncio.gather(*enrichment_tasks)
            valid_results = [r for r in batch_results if r]
            enriched_results.extend(valid_results)
            
            progress_bar.progress(batch_num / total_batches)
            st.write(f"✓ Processed batch {batch_num} of {total_batches} ({len(valid_results)} valid results)")
        
        self.results = enriched_results
        return enriched_results

    async def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict]:
        """Execute DuckDuckGo search with proper error handling"""
        try:
            results = []
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                results.append({
                    'title': result['title'],
                    'description': result['body'],
                    'url': result['link'],
                    'source': self._extract_source(result['link'])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []

    def _extract_source(self, url: str) -> str:
        """Extract the source platform from URL"""
        domain = urlparse(url).netloc.lower()
        if 'linkedin.com' in domain:
            return 'LinkedIn'
        elif 'wellfound.com' in domain:
            return 'WellFound'
        elif 'crunchbase.com' in domain:
            return 'Crunchbase'
        elif 'pitchbook.com' in domain:
            return 'PitchBook'
        return 'Other'

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results by domain"""
        seen_domains = set()
        unique_results = []
        
        for result in results:
            domain = urlparse(result['url']).netloc
            if domain not in seen_domains:
                seen_domains.add(domain)
                unique_results.append(result)
        
        return unique_results

    async def _validate_and_enrich(self, result: Dict, industry: str, stage: str) -> Optional[VCFirm]:
        """Use GPT-4 to validate and enrich VC data"""
        try:
            prompt = f"""
            Analyze this potential VC firm and extract relevant information:
            Title: {result['title']}
            Description: {result['description']}
            URL: {result['url']}
            Source: {result['source']}
            Industry Focus: {industry}
            Stage: {stage}

            Please validate if this is a legitimate VC firm and extract:
            1. Firm name
            2. Investment focus (especially regarding {industry})
            3. Stage preference (especially regarding {stage})
            """

            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
            )
            
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a VC analyst helping to validate and extract information about venture capital firms."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                
                # Extract information using regex or string parsing
                name_match = re.search(r"Firm name:?\s*(.+?)(?:\n|$)", content)
                focus_match = re.search(r"Investment focus:?\s*(.+?)(?:\n|$)", content)
                stage_match = re.search(r"Stage preference:?\s*(.+?)(?:\n|$)", content)
                
                if name_match:
                    return VCFirm(
                        name=name_match.group(1).strip(),
                        url=result['url'],
                        description=result['description'],
                        investment_focus=focus_match.group(1).strip() if focus_match else "",
                        stage_preference=stage_match.group(1).strip() if stage_match else "",
                        source=result['source']
                    )
            
            return None
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            st.error(f"Validation error: {str(e)}")
            return None

class VCEnrichmentAgent:
    """Agent responsible for enriching VC data with website information"""
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        )
        self.cache = {}
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
    async def enrich_vcs(self, vcs: List[VCFirm], industry: str):
        """Enrich VC firms with website data and relevance scores"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_context, limit=10)
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = []
            for vc in vcs:
                task = asyncio.create_task(self._enrich_vc(session, vc, industry))
                tasks.append(task)
            
            enriched_vcs = []
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if result:
                        enriched_vcs.append(result)
                except Exception as e:
                    logger.error(f"Error enriching VC: {str(e)}")
            
            return enriched_vcs
    
    async def _enrich_vc(self, session: aiohttp.ClientSession, vc: VCFirm, industry: str):
        """Enrich a single VC firm"""
        try:
            async with self.semaphore:
                if vc.url in self.cache:
                    logger.info(f"Using cached data for {vc.url}")
                    cached_data = self.cache[vc.url]
                    vc.description = cached_data.get('description', '')
                    vc.investment_focus = cached_data.get('investment_focus', '')
                    vc.relevance_score = cached_data.get('relevance_score', 0.0)
                    return vc
                
                content = await self._scrape_website(session, vc.url)
                if not content:
                    return None
                
                vc.description = content[:1000]  # Limit description length
                vc.investment_focus = await self._extract_investment_focus(content)
                vc.relevance_score = await self._calculate_relevance(content, industry)
                
                # Cache the results
                self.cache[vc.url] = {
                    'description': vc.description,
                    'investment_focus': vc.investment_focus,
                    'relevance_score': vc.relevance_score
                }
                
                return vc
        except Exception as e:
            logger.error(f"Error enriching {vc.url}: {str(e)}")
            return None
    
    async def _scrape_website(self, session: aiohttp.ClientSession, url: str) -> str:
        """Scrape website content with proper error handling and timeouts"""
        try:
            async with session.get(url, timeout=20) as response:
                if response.status != 200:
                    return ""
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit text length for processing
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""

    async def _extract_investment_focus(self, content: str) -> str:
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

    async def _calculate_relevance(self, vc_text: str, industry: str) -> float:
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
        self.cache = {}
        self.semaphore = asyncio.Semaphore(5)
        
    async def find_contacts(self, vcs: List[VCFirm]):
        """Find contact information for VC firms"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_context, limit=10)
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = []
            for vc in vcs:
                task = asyncio.create_task(self._find_vc_contacts(session, vc))
                tasks.append(task)
            
            for task in asyncio.as_completed(tasks):
                try:
                    await task
                except Exception as e:
                    logger.error(f"Error finding contacts: {str(e)}")
            
            return vcs
    
    async def _find_vc_contacts(self, session: aiohttp.ClientSession, vc: VCFirm):
        """Find contacts for a single VC firm"""
        try:
            async with self.semaphore:
                if vc.url in self.cache:
                    cached_data = self.cache[vc.url]
                    vc.emails = cached_data.get('emails', [])
                    vc.linkedin_profiles = cached_data.get('linkedin_profiles', [])
                    return
                
                contact_pages = await self._get_contact_pages(session, vc.url)
                emails = set()
                linkedin_profiles = set()
                
                for page_url in contact_pages:
                    try:
                        async with session.get(page_url, timeout=10) as response:
                            if response.status != 200:
                                continue
                            
                            content = await response.text()
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract emails
                            page_emails = self._extract_emails(content)
                            emails.update(page_emails)
                            
                            # Extract LinkedIn profiles
                            page_profiles = self._extract_linkedin_profiles(soup)
                            linkedin_profiles.update(page_profiles)
                    except Exception as e:
                        logger.error(f"Error processing contact page {page_url}: {str(e)}")
                
                vc.emails = list(emails)
                vc.linkedin_profiles = list(linkedin_profiles)
                
                # Cache the results
                self.cache[vc.url] = {
                    'emails': vc.emails,
                    'linkedin_profiles': vc.linkedin_profiles
                }
        except Exception as e:
            logger.error(f"Error finding contacts for {vc.url}: {str(e)}")
    
    async def _get_contact_pages(self, session: aiohttp.ClientSession, base_url: str) -> List[str]:
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
                if vc.stage_preference:
                    st.write("**Stage Preference:**", vc.stage_preference)
            
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
    st.title("🔍 Relevant VC Scraper")
    st.write("Find and analyze venture capital firms relevant to your startup")
    
    # Load startup data
    startup_data = load_startup_data()
    if not startup_data:
        st.error("❌ No startup data found. Please add your startup information first.")
        return
    
    # Display current startup data
    st.subheader("Current Startup Data")
    st.write(f"🏢 Industry: {startup_data['industry']}")
    st.write(f"📈 Stage: {startup_data['stage']}")
    st.write(f"📍 Location: {startup_data['location']}")
    
    if st.button("🔎 Find Relevant VCs"):
        search_agent = VCSearchAgent()
        enrichment_agent = VCEnrichmentAgent()
        contact_agent = VCContactAgent()
        
        try:
            # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the search process
            vcs = loop.run_until_complete(search_agent.search(
                industry=startup_data['industry'],
                stage=startup_data['stage']
            ))
            
            if vcs:
                st.success(f"✅ Found {len(vcs)} relevant VC firms!")
                display_results(vcs)
            else:
                st.warning("⚠️ No relevant VC firms found. Try adjusting your search criteria.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()
