import streamlit as st
from database import DatabaseManager
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import json
import re
from datetime import datetime
import time
import anthropic
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API clients
client = OpenAI()
claude = anthropic.Anthropic()

@st.cache_data(show_spinner=False)
def search_competitors(query, num_results=5):
    """Search for competitors using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return results

@st.cache_data(show_spinner=False)
def extract_website_info(url):
    """Extract information from a website."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text()
        
        # Basic cleaning
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        text_content = text_content[:5000]  # Limit length
        
        return text_content
    except Exception as e:
        st.error(f"Error extracting info from {url}: {str(e)}")
        return ""

@st.cache_data(show_spinner=False)
def analyze_competitor_with_ai(company_name, website_content):
    """Analyze competitor using AI."""
    try:
        # First try with Claude
        prompt = f"""Analyze this competitor for our market research:
        Company: {company_name}
        Website Content: {website_content}

        Provide a structured analysis with:
        1. Key Features/Products
        2. Target Market
        3. Strengths
        4. Weaknesses
        5. Potential Threats to Our Business
        6. Estimated Market Position (Enterprise/Mid-Market/SMB/Startup)
        7. Estimated Company Size
        8. Estimated Funding Range

        Format as JSON with these exact keys:
        features, target_market, strengths, weaknesses, threats, market_position, employee_count, funding
        """
        
        try:
            response = claude.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
        except:
            # Fallback to OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
        
        return result
    except Exception as e:
        st.error(f"Error analyzing with AI: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Online Competitor List", page_icon="🎯", layout="wide")
    
    # Initialize database connection
    db = DatabaseManager()
    
    # Sidebar navigation
    with st.sidebar:
        # Main navigation
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add a container for the rest of the sidebar space
        with st.container():
            # Add vertical space
            for _ in range(10):
                st.empty()
            
            # Startup selection at the bottom
            st.subheader("Startup Selection")
            startups = db.get_startups()
            if startups:
                startup_names = [s['name'] for s in startups]
                selected_startup_name = st.selectbox("Select Startup", startup_names, label_visibility="collapsed")
                
                # Create new startup button below the selection
                if st.button("Create New Startup"):
                    st.session_state.show_create_startup = True
            else:
                st.warning("No startups found. Create one first!")
                return
    
    # Get selected startup data
    selected_startup = next(s for s in startups if s['name'] == selected_startup_name)
    
    # Main content area
    st.title("Online Competitor List")
    st.caption("Step 1: Identify and track your online competitors")
    
    # Input section with improved layout
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add New Competitor")
            
            # Add tabs for manual and automatic competitor addition
            tab1, tab2 = st.tabs(["Manual Entry", "Auto-Discovery"])
            
            with tab1:
                with st.form("competitor_form"):
                    competitor_name = st.text_input("Competitor Name")
                    website = st.text_input("Website")
                    market_position = st.selectbox(
                        "Market Position",
                        ["Enterprise", "Mid-Market", "SMB", "Startup"]
                    )
                    funding = st.number_input("Total Funding (USD)", min_value=0, step=1000000)
                    employee_count = st.number_input("Employee Count", min_value=1, step=1)
                    
                    # Key features and strengths
                    features = st.text_area("Key Features (one per line)")
                    strengths = st.text_area("Strengths (one per line)")
                    weaknesses = st.text_area("Weaknesses (one per line)")
                    threats = st.text_area("Threats (one per line)")
                    
                    if st.form_submit_button("Add Competitor"):
                        if competitor_name and website:
                            competitor_data = {
                                'name': competitor_name,
                                'website': website,
                                'market_position': market_position,
                                'funding': funding,
                                'employee_count': employee_count,
                                'features': features.split('\n') if features else [],
                                'strengths': strengths.split('\n') if strengths else [],
                                'weaknesses': weaknesses.split('\n') if weaknesses else [],
                                'threats': threats.split('\n') if threats else [],
                                'startup_id': selected_startup['id']
                            }
                            
                            if db.add_competitor(competitor_data):
                                st.success(f"Added {competitor_name} to competitors list!")
                                st.rerun()
                            else:
                                st.error("Failed to add competitor")
                        else:
                            st.error("Name and website are required!")
            
            with tab2:
                search_query = st.text_input("Search Query", placeholder="Enter keywords to find competitors...")
                if st.button("Search"):
                    with st.spinner("Searching for competitors..."):
                        results = search_competitors(search_query)
                        
                        if results:
                            st.write("Found potential competitors:")
                            for result in results:
                                with st.expander(f"📊 {result['title'][:60]}..."):
                                    st.write(f"**URL:** {result['link']}")
                                    st.write(f"**Description:** {result['body']}")
                                    
                                    if st.button("Analyze", key=f"analyze_{result['link']}"):
                                        with st.spinner("Analyzing competitor..."):
                                            # Extract website content
                                            content = extract_website_info(result['link'])
                                            
                                            # Analyze with AI
                                            analysis = analyze_competitor_with_ai(result['title'], content)
                                            
                                            if analysis:
                                                # Prepare competitor data
                                                competitor_data = {
                                                    'name': result['title'][:100],
                                                    'website': result['link'],
                                                    'market_position': analysis.get('market_position', 'Unknown'),
                                                    'funding': float(re.sub(r'[^\d.]', '', str(analysis.get('funding', '0')))),
                                                    'employee_count': int(re.sub(r'[^\d]', '', str(analysis.get('employee_count', '1')))),
                                                    'features': analysis.get('features', []),
                                                    'strengths': analysis.get('strengths', []),
                                                    'weaknesses': analysis.get('weaknesses', []),
                                                    'threats': analysis.get('threats', []),
                                                    'startup_id': selected_startup['id']
                                                }
                                                
                                                # Add to database
                                                if db.add_competitor(competitor_data):
                                                    st.success(f"Added {result['title'][:60]} to competitors list!")
                                                    time.sleep(1)
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to add competitor")
                                        
                        else:
                            st.warning("No results found. Try different keywords.")
        
        with col2:
            st.subheader("Competitor Analysis")
            competitors = db.get_competitors(selected_startup['id'])
            
            if competitors:
                # Create metrics
                total_competitors = len(competitors)
                avg_funding = np.mean([c['funding'] for c in competitors])
                avg_employees = np.mean([c['employee_count'] for c in competitors])
                
                # Display metrics in columns
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Competitors", total_competitors)
                m2.metric("Avg. Funding", f"${avg_funding:,.0f}")
                m3.metric("Avg. Employees", f"{avg_employees:.0f}")
                
                # Display competitor table
                df = pd.DataFrame(competitors)
                st.dataframe(
                    df[['name', 'market_position', 'funding', 'employee_count', 'website']],
                    column_config={
                        'name': 'Competitor',
                        'market_position': 'Market Position',
                        'funding': st.column_config.NumberColumn(
                            'Funding',
                            format='$%d'
                        ),
                        'employee_count': 'Employees',
                        'website': 'Website'
                    },
                    hide_index=True
                )
                
                # Detailed competitor view
                st.subheader("Detailed Analysis")
                for competitor in competitors:
                    with st.expander(f"🔍 {competitor['name']}"):
                        # Market Position and Competitive Advantage
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Market Position", competitor['market_position'])
                        with col4:
                            st.metric("Employee Count", competitor['employee_count'])
                        
                        # Features and Analysis
                        st.subheader("Key Features")
                        for feature in competitor['features']:
                            st.markdown(f"• {feature}")
                        
                        # SWOT Analysis
                        swot1, swot2 = st.columns(2)
                        
                        with swot1:
                            st.subheader("Strengths")
                            for strength in competitor['strengths']:
                                st.markdown(f"• {strength}")
                            
                            st.subheader("Weaknesses")
                            for weakness in competitor['weaknesses']:
                                st.markdown(f"• {weakness}")
                        
                        with swot2:
                            st.subheader("Threats to Our Business")
                            for threat in competitor['threats']:
                                st.markdown(f"• {threat}")
            else:
                st.info("No competitors added yet. Add your first competitor!")

if __name__ == "__main__":
    main()
