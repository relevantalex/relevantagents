import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
from io import StringIO
import csv
from urllib.parse import urlparse
import re
from database import DatabaseManager
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VCScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))

    def extract_linkedin_profiles(self, soup) -> List[str]:
        """Extract LinkedIn profile URLs from HTML"""
        linkedin_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'linkedin.com/in/' in href:
                linkedin_links.append(href)
        return list(set(linkedin_links))

    def scrape_website(self, url: str) -> Dict[str, any]:
        """Scrape a website for contact information"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content
            text = soup.get_text()
            
            # Get emails and LinkedIn profiles
            emails = self.extract_emails(text)
            linkedin_profiles = self.extract_linkedin_profiles(soup)
            
            return {
                'url': url,
                'emails': emails,
                'linkedin_profiles': linkedin_profiles,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'emails': [],
                'linkedin_profiles': [],
                'status': f'error: {str(e)}'
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

def process_vc_list(uploaded_file, startup_data: Dict[str, any]):
    """Process list of VCs from uploaded file"""
    vc_scraper = VCScraper()
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
                
                status_text.text(f"Processing {name}...")
                
                # Skip if no website provided
                if not website or pd.isna(website):
                    continue
                
                # Add protocol if missing
                if not website.startswith(('http://', 'https://')):
                    website = 'https://' + website
                
                # Scrape website
                scrape_result = vc_scraper.scrape_website(website)
                
                results.append({
                    'name': name,
                    'website': website,
                    'emails': scrape_result['emails'],
                    'linkedin_profiles': scrape_result['linkedin_profiles'],
                    'status': scrape_result['status']
                })
                
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                results.append({
                    'name': name,
                    'website': website,
                    'emails': [],
                    'linkedin_profiles': [],
                    'status': f'error: {str(e)}'
                })
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if results:
            st.subheader("Results")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Display success/error counts
            success_count = sum(1 for r in results if r['status'] == 'success')
            st.write(f"Successfully processed {success_count} out of {len(results)} VCs")
            
            # Show results in an expandable section
            with st.expander("View Detailed Results", expanded=True):
                st.dataframe(
                    results_df,
                    column_config={
                        "name": "VC Name",
                        "website": "Website",
                        "emails": "Found Emails",
                        "linkedin_profiles": "LinkedIn Profiles",
                        "status": "Status"
                    }
                )
            
            # Download results button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results as CSV",
                csv,
                "vc_scraping_results.csv",
                "text/csv",
                key='download-csv'
            )

def main():
    st.set_page_config(
        page_title="VC Email List Scraper",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 VC Email List Scraper")
    st.caption("Upload a list of VCs to find their contact information")
    
    # Load startup data
    startup_data = load_startup_data()
    if not startup_data:
        return
    
    # File upload section
    st.subheader("Upload VC List")
    st.info("""
    Please upload a CSV file with the following columns:
    - name: Name of the VC firm
    - website: Website URL of the VC firm
    
    The scraper will process the first 20 VCs in the list.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with VC names and websites"
    )
    
    if uploaded_file:
        process_vc_list(uploaded_file, startup_data)
    else:
        st.write("No file uploaded yet.")
        
        # Show example CSV structure
        st.subheader("Example CSV Structure")
        example_df = pd.DataFrame([
            {"name": "Example VC", "website": "www.examplevc.com"},
            {"name": "Another VC", "website": "www.anothervc.com"}
        ])
        st.dataframe(example_df)

if __name__ == "__main__":
    main()
