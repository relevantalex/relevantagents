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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

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

def clean_json_response(response: str) -> str:
    """Clean AI response to ensure valid JSON"""
    # Find the first '[' and last ']'
    start = response.find('[')
    end = response.rfind(']')
    
    if start == -1 or end == -1:
        raise ValueError("No valid JSON array found in response")
        
    json_str = response[start:end + 1]
    # Remove any markdown formatting
    json_str = re.sub(r'```json\s*|\s*```', '', json_str)
    return json_str

@st.cache_data(ttl=3600, show_spinner=False)
def identify_industries(pitch: str) -> List[str]:
    """Identify potential industries based on the pitch using AI"""
    ai = AIProvider()
    prompt = f"""Based on this pitch: "{pitch}"
    Identify exactly 3 specific, relevant industries or market segments.
    Format your response as a JSON array with exactly 3 strings.
    Make each industry name specific and descriptive.
    Example: ["AI-Powered Security Analytics", "Retail Technology Solutions", "Computer Vision SaaS"]
    
    Return ONLY the JSON array, no other text."""

    try:
        response = ai.generate_response(prompt)
        cleaned_response = clean_json_response(response)
        industries = json.loads(cleaned_response)
        return industries[:3]
    except Exception as e:
        logger.error(f"Industry identification failed: {str(e)}")
        return ["Technology Solutions", "Software Services", "Digital Innovation"]

@dataclass
class CompetitorData:
    domain: str
    name: str
    website: str
    description: str
    differentiator: str
    relevance_score: float
    confidence_score: float

class SearchProvider:
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        raise NotImplementedError

class BraveSearch(SearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        headers = {"X-Subscription-Token": self.api_key}
        params = {
            "q": query,
            "count": max_results,
            "search_lang": "en"
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for web in data.get("web", {}).get("results", []):
                results.append({
                    "title": web.get("title", ""),
                    "link": web.get("url", ""),
                    "description": web.get("description", "")
                })
            return results
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            return []

class DuckDuckGoSearch(SearchProvider):
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "description": r.get("body", "")
                    })
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []

@lru_cache(maxsize=100)
def scrape_website_content(url: str) -> Optional[Dict[str, str]]:
    """Enhanced website scraping with better content extraction"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        # Get meta descriptions
        meta_desc = ""
        meta_tags = [
            soup.find("meta", attrs={"name": "description"}),
            soup.find("meta", attrs={"property": "og:description"}),
            soup.find("meta", attrs={"name": "twitter:description"})
        ]
        for tag in meta_tags:
            if tag and tag.get("content"):
                meta_desc = tag.get("content")
                break
        
        # Get title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.string
        
        # Try to get main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '#main-content', '.main-content', '[class*="content-main"]']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            # If no main content area found, look for largest text block
            text_blocks = []
            for p in soup.find_all(['p', 'div']):
                text = p.get_text(strip=True)
                if len(text) > 50:  # Minimum length to consider
                    text_blocks.append((len(text), p))
            
            if text_blocks:
                text_blocks.sort(reverse=True)
                main_content = text_blocks[0][1]
        
        # Get content
        content = ""
        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            content = soup.get_text(separator=' ', strip=True)
        
        # Clean content
        content = re.sub(r'\s+', ' ', content)
        content = content[:10000]  # Limit length
        
        # Extract company-specific information
        about_links = soup.find_all('a', href=re.compile(r'about|company|team', re.I))
        about_content = ""
        if about_links:
            about_sections = soup.find_all(['div', 'section'], 
                                        class_=re.compile(r'about|company|team', re.I))
            if about_sections:
                about_content = ' '.join(s.get_text(separator=' ', strip=True) 
                                      for s in about_sections)
        
        return {
            "title": title,
            "meta_description": meta_desc,
            "content": content,
            "about_content": about_content
        }
        
    except Exception as e:
        logger.error(f"Failed to scrape website {url}: {str(e)}")
        return None

def is_news_site(website_data: Dict[str, str]) -> bool:
    """Check if the website is a news outlet"""
    news_indicators = [
        'news', 'magazine', 'media', 'press', 'journal', 'blog', 'post',
        'article', 'editorial', 'newsletter', 'daily', 'weekly', 'tribune',
        'times', 'gazette', 'herald', 'chronicle', 'observer', 'digest',
        'report', 'wire', 'feed', 'bulletin'
    ]
    
    # Check domain and content for news indicators
    text_to_check = f"{website_data['title']} {website_data['meta_description']}".lower()
    
    # Check for news-related words
    for indicator in news_indicators:
        if indicator in text_to_check:
            return True
            
    # Check for typical news site patterns
    if re.search(r'breaking\s+news|latest\s+news|top\s+stories', website_data['content'].lower()):
        return True
        
    return False

def validate_competitor(website_data: Dict[str, str], industry: str, startup_pitch: str) -> Tuple[bool, str, str, float]:
    """Validate competitor with multiple checks and scoring"""
    ai = AIProvider()
    
    validation_prompt = f"""Analyze this potential competitor:
    Title: {website_data['title']}
    Description: {website_data['meta_description']}
    Website Content: {website_data['content'][:2000]}
    
    Target Industry: {industry}
    Our Startup's Pitch: {startup_pitch}
    
    Perform a detailed analysis and return a JSON object with:
    1. is_company: Is this a real company (not news/blog/etc)?
    2. is_competitor: Do they compete in our industry?
    3. company_description: Detailed 2-sentence description of their product/service
    4. differentiator: Their key advantage or unique selling point
    5. relevance_score: 0-100 rating as competitor
    6. confidence_score: 0-100 rating of analysis confidence
    7. reasoning: Brief explanation of the scoring
    
    Return ONLY the JSON object, no other text."""
    
    try:
        response = ai.generate_response(validation_prompt)
        validation = json.loads(clean_json_response(response))
        
        is_valid = (
            validation['is_company'] and 
            validation['is_competitor'] and 
            validation['relevance_score'] >= 70 and 
            validation['confidence_score'] >= 60
        )
        
        return (
            is_valid,
            validation['company_description'],
            validation['differentiator'],
            validation['relevance_score']
        )
    except Exception as e:
        logger.error(f"Competitor validation failed: {str(e)}")
        return (False, "", "", 0)

def find_competitors(industry: str, pitch: str, progress_bar) -> List[Dict]:
    """Find competitors using multiple search providers and validation methods"""
    ai = AIProvider()
    all_competitors: Set[str] = set()
    potential_competitors: List[CompetitorData] = []
    
    # Initialize search providers
    search_providers = [
        BraveSearch(api_key="BSAFYF-wl3SieZb-w4E18vNNwXldlnH"),
        DuckDuckGoSearch()
    ]
    
    try:
        progress_bar.progress(0.1, "🔍 Generating search queries...")
        search_prompt = f"""For a startup in {industry} with this pitch: "{pitch}"
        Create 4 different search queries to find direct competitors.
        Make them specific and targeted to find actual companies.
        Include company-specific terms like "company", "platform", "solution", "software".
        Return a JSON array of queries.
        Return ONLY the JSON array, no other text."""

        queries = json.loads(clean_json_response(ai.generate_response(search_prompt)))
        
        total_steps = len(search_providers) * len(queries)
        current_step = 0
        
        for provider in search_providers:
            for query in queries:
                current_step += 1
                progress = 0.2 + (0.6 * current_step / total_steps)
                progress_bar.progress(progress, f"🌐 Searching with provider {type(provider).__name__} ({current_step}/{total_steps})")
                
                results = provider.search(query, max_results=5)
                
                for result in results:
                    try:
                        domain = urlparse(result['link']).netloc
                        if not domain or domain in all_competitors:
                            continue
                            
                        all_competitors.add(domain)
                        
                        # Scrape website
                        website_data = scrape_website_content(result['link'])
                        if not website_data:
                            continue
                            
                        # Skip news sites
                        if is_news_site(website_data):
                            continue
                            
                        # Validate competitor
                        is_valid, description, differentiator, relevance = validate_competitor(
                            website_data, industry, pitch
                        )
                        
                        if is_valid:
                            competitor = CompetitorData(
                                domain=domain,
                                name=website_data['title'].split('|')[0].strip(),
                                website=result['link'],
                                description=description,
                                differentiator=differentiator,
                                relevance_score=relevance,
                                confidence_score=1.0  # Base confidence
                            )
                            potential_competitors.append(competitor)
                            
                    except Exception as e:
                        logger.error(f"Error processing result: {str(e)}")
                        continue
        
        # Sort by relevance and convert to final format
        progress_bar.progress(0.9, "🏆 Selecting top competitors...")
        potential_competitors.sort(key=lambda x: x.relevance_score, reverse=True)
        
        final_competitors = [
            {
                "name": comp.name,
                "website": comp.website,
                "description": comp.description,
                "differentiator": comp.differentiator
            }
            for comp in potential_competitors[:3]
        ]
        
        progress_bar.progress(1.0, "✅ Analysis complete!")
        return final_competitors
        
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
        try:
            # Create a container for the analysis process
            analysis_container = st.container()
            with analysis_container:
                st.markdown("### 🤖 AI Market Analysis in Progress")
                
                # Industry Analysis
                with st.status("🎯 Phase 1: Industry Analysis", expanded=True) as status:
                    st.write("Analyzing your startup's market positioning...")
                    industries = identify_industries(selected_startup.get('pitch', ''))
                    st.session_state.industries = industries
                    st.session_state.competitors = {}
                    status.update(label="✅ Phase 1: Industry Analysis - Complete", state="complete")
                    
                    # Show identified industries
                    st.success("Identified Target Industries:")
                    for idx, industry in enumerate(industries, 1):
                        st.markdown(f"**{idx}.** {industry}")
                
                # Competitor Analysis
                with st.status("🔍 Phase 2: Competitor Analysis", expanded=True) as status:
                    # Create progress bars for each industry
                    progress_bars = {}
                    
                    for industry in industries:
                        st.write(f"\n**Analyzing {industry}**")
                        progress_bars[industry] = st.progress(0, f"Starting analysis for {industry}...")
                        
                        competitors = find_competitors(industry, selected_startup.get('pitch', ''), progress_bars[industry])
                        st.session_state.competitors[industry] = competitors
                    
                    status.update(label="✅ Phase 2: Competitor Analysis - Complete", state="complete")
                
                st.success("🎉 Market Analysis Successfully Completed!")
                time.sleep(1)  # Brief pause for visual feedback
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
