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

def process_vc_list(uploaded_file, startup_pitch: str) -> pd.DataFrame:
    if not startup_pitch.strip():
        st.error("Please enter your startup pitch first!")
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['name', 'website']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain 'name' and 'website' columns!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()

    # Process only first 20 VCs
    df = df.head(20)
    total_vcs = len(df)
    
    results = []
    vc_scraper = VCScraper()
    vc_analyzer = VCAnalyzer()
    
    # Create placeholder for results table
    results_placeholder = st.empty()
    
    # Create columns for progress information
    col1, col2 = st.columns([3, 1])
    with col1:
        progress_bar = st.progress(0)
    with col2:
        counter = st.empty()
    
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        try:
            progress = (idx + 1) / total_vcs
            progress_bar.progress(progress)
            counter.write(f"{idx + 1}/{total_vcs}")
            
            name = row['name']
            website = str(row['website']).strip()
            
            status_text.write(f"🔍 Analyzing: {name}")
            
            if not website or website.lower() == 'nan':
                continue
                
            # Scrape website
            website_data = vc_scraper.scrape_website(website)
            
            if website_data['text']:
                # Analyze VC fit
                analysis = vc_analyzer.analyze_vc_fit({
                    'text': website_data['text'],
                    'portfolio': website_data.get('portfolio', '')
                }, startup_pitch)
                
                result = {
                    'VC Firm Name': name,
                    'Contact Email': ', '.join(website_data['emails']) if website_data['emails'] else 'No email found',
                    'Why This Fund is a Match': analysis.get('match_reason', '') if analysis.get('match_score', 0) >= 70 else 'Not a strong match',
                    'Partner LinkedIn': website_data.get('partner', 'Not found'),
                    'Analyst LinkedIn': website_data.get('analyst', 'Not found'),
                    'Investment Verticals': ', '.join(analysis.get('verticals', [])),
                    'Investment Stages': ', '.join(analysis.get('stages', [])),
                    'Typical Check Size': analysis.get('check_size', 'Unknown'),
                    'Geographic Focus': analysis.get('geography', 'Unknown'),
                    'Similar Portfolio Companies': ', '.join(analysis.get('similar_companies', [])),
                    'Match Score': analysis.get('match_score', 0)
                }
                
                results.append(result)
                
                # Update results table
                temp_df = pd.DataFrame(results)
                results_placeholder.dataframe(temp_df)
                
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    counter.empty()
    status_text.empty()
    
    final_df = pd.DataFrame(results)
    
    if final_df.empty:
        st.warning("No results found. Try adjusting your startup pitch or check the VC list.")
    else:
        st.success(f"✅ Analysis complete! Found {len(final_df)} matching VCs.")
    
    return final_df

def main():
    st.title("🎯 VC Analysis")
    
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        1. Enter your startup pitch, including:
           - Your value proposition
           - Target market
           - Current traction
           - Funding stage
        2. Upload a CSV file with VC information (columns: name, website)
        3. Click 'Start VC Analysis' to begin
        
        The analysis will:
        - Scrape VC websites for contact information
        - Analyze investment focus
        - Evaluate startup-VC fit
        - Find team member profiles
        """)
    
    if 'startup_pitch' not in st.session_state:
        st.session_state.startup_pitch = ""
    
    with st.expander("📝 Enter Your Startup Pitch", expanded=not bool(st.session_state.startup_pitch)):
        startup_pitch = st.text_area(
            "Describe your startup, target market, and unique value proposition",
            value=st.session_state.startup_pitch,
            height=200,
            placeholder="Example: We are a B2B SaaS platform using AI to automate financial reporting..."
        )
        if startup_pitch != st.session_state.startup_pitch:
            st.session_state.startup_pitch = startup_pitch
    
    uploaded_file = st.file_uploader(
        "Upload your VC list (CSV with columns: name, website)",
        type=['csv']
    )
    
    if uploaded_file and st.session_state.startup_pitch:
        if st.button("🚀 Start VC Analysis"):
            with st.spinner("Analyzing VCs..."):
                results_df = process_vc_list(uploaded_file, st.session_state.startup_pitch)
                
                if not results_df.empty:
                    # Create download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "vc_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
    else:
        st.info("👆 Please upload a CSV file with VC information and provide your startup pitch to begin the analysis.")

if __name__ == "__main__":
    main()
