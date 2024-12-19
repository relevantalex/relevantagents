import asyncio
import aiohttp
import json
import streamlit as st

async def test_google_search(query: str):
    """Test Google Custom Search API"""
    try:
        # Replace these with your actual API keys
        google_api_key = st.secrets["api_keys"]["google_api_key"]
        google_cx = st.secrets["api_keys"]["google_cx"]
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': google_api_key,
            'cx': google_cx,
            'q': query
        }
        
        print(f"\nTesting search with query: {query}")
        print("API URL:", url)
        print("Parameters:", json.dumps(params, indent=2))
        
        async with aiohttp.ClientSession() as session:
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
