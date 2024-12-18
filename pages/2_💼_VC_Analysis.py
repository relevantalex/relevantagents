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
from duckduckgo_search import DDGS
import json

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
                            additional_info['team_members'].append(result['title'])
                        elif 'investments' in query:
                            additional_info['recent_investments'].append(result['body'])
            except Exception as e:
                logger.error(f"Error enriching VC data: {str(e)}")
        
        return {**vc_data, **additional_info}

class VCScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_emails(self, text: str) -> List[str]:
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, text)))
        # Filter out common false positives
        filtered_emails = [
            email for email in emails 
            if not any(exclude in email.lower() for exclude in ['example.com', 'domain.com'])
        ]
        return filtered_emails

    def extract_linkedin_profiles(self, soup) -> Dict[str, str]:
        profiles = {
            'partner': '',
            'analyst': ''
        }
        
        # Look for team/about section
        team_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
            keyword in x.lower() for keyword in ['team', 'people', 'about', 'members']
        ))
        
        linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?'
        
        for section in team_sections:
            text = section.get_text().lower()
            links = section.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                if re.match(linkedin_pattern, href):
                    surrounding_text = link.get_text().lower() + ' ' + text[:100]
                    
                    # Check for partner/senior roles
                    if not profiles['partner'] and any(role in surrounding_text for role in 
                        ['partner', 'managing director', 'founder', 'principal', 'head']):
                        profiles['partner'] = href
                    
                    # Check for analyst/associate roles
                    elif not profiles['analyst'] and any(role in surrounding_text for role in 
                        ['analyst', 'associate', 'investment professional']):
                        profiles['analyst'] = href
                        
                if profiles['partner'] and profiles['analyst']:
                    break
                    
        return profiles

    def scrape_website(self, url: str) -> Dict[str, any]:
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer']):
                element.decompose()

            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main'])
            text = main_content.get_text() if main_content else soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            
            # Extract emails
            emails = self.extract_emails(text)
            
            # Extract LinkedIn profiles
            linkedin_profiles = self.extract_linkedin_profiles(soup)
            
            # Try to find portfolio companies
            portfolio_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
                keyword in (x.lower() if x else '') for keyword in ['portfolio', 'companies', 'investments']
            ))
            
            portfolio_text = ''
            for section in portfolio_sections:
                portfolio_text += section.get_text() + '\n'
            
            # Try to find investment focus/thesis
            focus_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
                keyword in (x.lower() if x else '') for keyword in ['focus', 'thesis', 'strategy', 'about']
            ))
            
            focus_text = ''
            for section in focus_sections:
                focus_text += section.get_text() + '\n'
            
            return {
                'text': text[:5000],  # Main content
                'emails': emails,
                'portfolio': portfolio_text[:2000],  # Portfolio companies
                'focus': focus_text[:2000],  # Investment focus
                'partner': linkedin_profiles['partner'],
                'analyst': linkedin_profiles['analyst']
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'text': '',
                'emails': [],
                'portfolio': '',
                'focus': '',
                'partner': '',
                'analyst': ''
            }

class VCAnalyzer:
    def __init__(self):
        self.ai_provider = "openai"  # Default to OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("api_keys", {}).get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")
        self.model = "gpt-4-turbo-preview"

    def analyze_vc_fit(self, website_data: Dict[str, str], startup_pitch: str) -> Dict[str, any]:
        prompt = f"""
        Analyze this VC firm's fit for a startup based on the following information.
        
        VC Website Content:
        {website_data.get('text', '')}
        
        VC Portfolio Companies:
        {website_data.get('portfolio', '')}
        
        VC Investment Focus:
        {website_data.get('focus', '')}
        
        Startup Pitch:
        {startup_pitch}
        
        Please analyze and provide:
        1. A list of key investment verticals/sectors
        2. Investment stage preferences
        3. Typical check size (if mentioned)
        4. Geographic focus
        5. Notable portfolio companies that are similar to the startup
        6. A match score (0-100) based on:
           - Sector alignment (40%)
           - Stage fit (30%)
           - Geographic fit (15%)
           - Portfolio synergy (15%)
        7. If score >= 70, provide a 2-3 sentence explanation of why this fund is a good match
        
        Format the response as JSON with keys: 
        verticals, stages, check_size, geography, similar_companies, match_score, match_reason
        
        Keep the response concise and focused on factual information from the provided text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert VC analyst who helps startups identify the best matching venture capital firms."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.5,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            try:
                result = eval(content)
                return result
            except:
                # If eval fails, try to extract just the match score and reason
                match_score = 0
                match_reason = "Unable to determine match"
                
                if '"match_score":' in content:
                    try:
                        match_score = int(re.search(r'"match_score":\s*(\d+)', content).group(1))
                    except:
                        pass
                        
                if '"match_reason":' in content:
                    try:
                        match_reason = re.search(r'"match_reason":\s*"([^"]+)"', content).group(1)
                    except:
                        pass
                
                return {
                    'verticals': [],
                    'stages': [],
                    'check_size': 'Unknown',
                    'geography': 'Unknown',
                    'similar_companies': [],
                    'match_score': match_score,
                    'match_reason': match_reason
                }
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                'verticals': [],
                'stages': [],
                'check_size': 'Unknown',
                'geography': 'Unknown',
                'similar_companies': [],
                'match_score': 0,
                'match_reason': f"Error analyzing VC fit: {str(e)}"
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
    
    # Get startup documents
    documents = db.get_documents(st.session_state.selected_startup['id'])
    
    # Prepare startup data
    startup_data = {
        'name': st.session_state.selected_startup['name'],
        'pitch': st.session_state.selected_startup.get('pitch', ''),
        'industry': '',  # Will be extracted from documents
        'stage': '',     # Will be extracted from documents
        'location': '',  # Will be extracted from documents
        'documents': documents
    }
    
    # Try to extract additional information from documents
    for doc in documents:
        content = doc.get('content', '')
        if not content:
            continue
            
        try:
            doc_data = json.loads(content)
            
            # Extract industry information
            if doc.get('type') == 'industry_analysis':
                startup_data['industry'] = doc_data.get('industry', '')
            
            # Extract stage information
            if doc.get('type') == 'company_info':
                startup_data['stage'] = doc_data.get('stage', '')
                startup_data['location'] = doc_data.get('location', '')
                
        except json.JSONDecodeError:
            continue
    
    return startup_data

def process_vc_list(uploaded_file, startup_data: Dict[str, any]) -> pd.DataFrame:
    """Process list of VCs from uploaded file"""
    vc_scraper = VCScraper()
    vc_analyzer = VCAnalyzer()
    vc_researcher = VCResearchEngine()
    results = []
    
    # Process uploaded VCs
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.head(20)  # Process first 20 VCs
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            try:
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                
                name = row['name']
                website = str(row['website']).strip()
                
                status_text.write(f"🔍 Analyzing {name}...")
                
                if not website or website.lower() == 'nan':
                    continue
                
                # Basic website scraping
                website_data = vc_scraper.scrape_website(website)
                
                # Enrich with additional data
                enriched_data = vc_researcher.enrich_vc_data({
                    'name': name,
                    'website': website,
                    **website_data
                })
                
                # Analyze VC fit
                analysis = vc_analyzer.analyze_vc_fit(enriched_data, startup_data)
                
                if analysis['match_score'] >= 50:  # Include more VCs in results
                    results.append({
                        'Source': 'Uploaded List',
                        'VC Firm Name': name,
                        'Contact Email': ', '.join(website_data['emails']) if website_data['emails'] else 'No email found',
                        'Why This Fund is a Match': analysis['match_reason'],
                        'Partner LinkedIn': website_data.get('partner', 'Not found'),
                        'Analyst LinkedIn': website_data.get('analyst', 'Not found'),
                        'Investment Verticals': ', '.join(analysis.get('verticals', [])),
                        'Investment Stages': ', '.join(analysis.get('stages', [])),
                        'Typical Check Size': analysis.get('check_size', 'Unknown'),
                        'Geographic Focus': analysis.get('geography', 'Unknown'),
                        'Similar Portfolio Companies': ', '.join(analysis.get('similar_companies', [])),
                        'Match Score': analysis.get('match_score', 0)
                    })
                
                # Update results in real-time
                temp_df = pd.DataFrame(results)
                st.dataframe(temp_df)
                
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
    
    return pd.DataFrame(results)

def find_additional_vcs(startup_data: Dict[str, any]) -> pd.DataFrame:
    """Find additional relevant VCs through internet research"""
    vc_researcher = VCResearchEngine()
    vc_scraper = VCScraper()
    vc_analyzer = VCAnalyzer()
    results = []
    
    st.write("🔎 Searching for additional relevant VCs...")
    
    # Search for relevant VCs
    discovered_vcs = vc_researcher.search_vcs(startup_data)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, vc in enumerate(discovered_vcs):
        try:
            progress = (idx + 1) / len(discovered_vcs)
            progress_bar.progress(progress)
            
            status_text.write(f"🔍 Analyzing discovered VC: {vc['name']}...")
            
            # Scrape website
            website_data = vc_scraper.scrape_website(vc['website'])
            
            # Enrich with additional data
            enriched_data = vc_researcher.enrich_vc_data({
                **vc,
                **website_data
            })
            
            # Analyze VC fit
            analysis = vc_analyzer.analyze_vc_fit(enriched_data, startup_data)
            
            if analysis['match_score'] >= 60:  # Higher threshold for discovered VCs
                results.append({
                    'Source': 'Discovered',
                    'VC Firm Name': vc['name'],
                    'Contact Email': ', '.join(website_data['emails']) if website_data['emails'] else 'No email found',
                    'Why This Fund is a Match': analysis['match_reason'],
                    'Partner LinkedIn': website_data.get('partner', 'Not found'),
                    'Analyst LinkedIn': website_data.get('analyst', 'Not found'),
                    'Investment Verticals': ', '.join(analysis.get('verticals', [])),
                    'Investment Stages': ', '.join(analysis.get('stages', [])),
                    'Typical Check Size': analysis.get('check_size', 'Unknown'),
                    'Geographic Focus': analysis.get('geography', 'Unknown'),
                    'Similar Portfolio Companies': ', '.join(analysis.get('similar_companies', [])),
                    'Match Score': analysis.get('match_score', 0)
                })
                
                # Update results in real-time
                temp_df = pd.DataFrame(results)
                st.dataframe(temp_df)
                
        except Exception as e:
            logger.error(f"Error processing discovered VC {vc['name']}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    st.title("🎯 VC Analysis")
    st.caption("Find and analyze potential venture capital investors")
    
    # Load startup data from database
    startup_data = load_startup_data()
    if not startup_data:
        st.stop()
    
    # Display startup information
    with st.expander("📋 Current Startup Information", expanded=False):
        st.markdown(f"""
        ### {startup_data['name']}
        
        **Pitch**: {startup_data['pitch']}
        
        **Industry**: {startup_data['industry'] or 'Not specified'}
        **Stage**: {startup_data['stage'] or 'Not specified'}
        **Location**: {startup_data['location'] or 'Not specified'}
        """)
        
        if not startup_data['industry'] or not startup_data['stage']:
            st.warning("⚠️ Some startup information is missing. For better VC matching, please complete your startup profile in the Startup Manager.")
    
    # File uploader for VC list
    uploaded_file = st.file_uploader(
        "Upload your VC list (CSV with columns: name, website)",
        type=['csv']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_uploaded = st.button("🔍 Analyze Uploaded VCs")
    with col2:
        discover_new = st.button("🌐 Discover New VCs")
    
    if analyze_uploaded and uploaded_file:
        with st.spinner("Analyzing uploaded VCs..."):
            results_df = process_vc_list(uploaded_file, startup_data)
            
            if not results_df.empty:
                st.success(f"✅ Found {len(results_df)} matching VCs from your list!")
                
                # Create download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "vc_analysis_results.csv",
                    "text/csv",
                    key='download-csv-uploaded'
                )
    
    if discover_new:
        with st.spinner("Discovering new VCs..."):
            discovered_df = find_additional_vcs(startup_data)
            
            if not discovered_df.empty:
                st.success(f"✅ Discovered {len(discovered_df)} new matching VCs!")
                
                # Create download button
                csv = discovered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Discovered VCs",
                    csv,
                    "discovered_vcs.csv",
                    "text/csv",
                    key='download-csv-discovered'
                )

if __name__ == "__main__":
    main()
