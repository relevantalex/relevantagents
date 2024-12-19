import asyncio
import aiohttp
import json
import ssl
import certifi
import os
import streamlit as st

# Get API keys from environment or secrets
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["api_keys"]["google_api_key"]
    GOOGLE_CX = os.getenv("GOOGLE_CX") or st.secrets["api_keys"]["google_cx"]
except Exception as e:
    raise ValueError("Please set GOOGLE_API_KEY and GOOGLE_CX in environment variables or Streamlit secrets") from e

async def test_google_search(query: str):
    """Test Google Custom Search API"""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CX,
            'q': query
        }
        
        print(f"\nTesting search with query: {query}")
        print("API URL:", url)
        print("Parameters:", json.dumps({**params, 'key': '[REDACTED]', 'cx': '[REDACTED]'}, indent=2))
        
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.get(url, params=params) as response:
                print(f"\nResponse Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    print(f"Found {len(items)} results")
                    
                    for i, item in enumerate(items[:3], 1):
                        print(f"\nResult {i}:")
                        print(f"Title: {item['title']}")
                        print(f"URL: {item['link']}")
                        print(f"Snippet: {item.get('snippet', 'No snippet available')}")
                else:
                    error_text = await response.text()
                    print("Error Response:", error_text)
    except Exception as e:
        print(f"Error: {str(e)}")

async def main():
    # Test queries
    queries = [
        'site:linkedin.com/company "venture capital" "healthcare" "medical"',
        'site:crunchbase.com/organization "venture capital" "healthcare"'
    ]
    
    for query in queries:
        await test_google_search(query)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
