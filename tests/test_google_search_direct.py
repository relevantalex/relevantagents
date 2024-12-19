import os
import asyncio
import aiohttp
import ssl
import certifi
import json
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API keys
os.environ['GOOGLE_API_KEY'] = 'dummy_google_api_key'
os.environ['GOOGLE_CX'] = 'dummy_google_cx'

async def google_search(query: str, max_results: int = 5) -> List[Dict]:
    """Execute Google Custom Search with proper error handling"""
    results = []
    
    # Get API credentials
    api_key = os.environ['GOOGLE_API_KEY']
    cx = os.environ['GOOGLE_CX']
    
    if not api_key or not cx:
        raise ValueError("Google Search API credentials not found in environment variables")
    
    # Create SSL context for API requests
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_context)
    
    # Add headers including referer
    headers = {
        'Referer': 'http://localhost:8501',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
            for start in range(1, min(max_results + 1, 101), 10):
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': api_key,
                    'cx': cx,
                    'q': query,
                    'start': start
                }
                
                logger.info(f"Making request to Google Custom Search API...")
                logger.info(f"Query: {query}")
                logger.info(f"Start: {start}")
                
                async with session.get(url, params=params) as response:
                    response_text = await response.text()
                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response headers: {response.headers}")
                    logger.info(f"Response body: {response_text}")
                    
                    if response.status == 200:
                        data = json.loads(response_text)
                        items = data.get('items', [])
                        logger.info(f"Found {len(items)} results")
                        
                        for item in items:
                            result = {
                                'title': item['title'],
                                'description': item.get('snippet', ''),
                                'url': item['link']
                            }
                            results.append(result)
                            logger.info(f"Added result: {result['title']}")
                    
                    elif response.status == 429:  # Rate limit
                        logger.warning("Rate limit reached. Waiting before retrying...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        logger.error(f"API Error (Status {response.status}): {response_text}")
                        break
                
                if not items:  # No more results
                    break
                
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise
    
    return results

async def main():
    # Test search query
    query = 'site:linkedin.com/company "venture capital" "healthcare" "medical" "biotech" "seed"'
    
    try:
        results = await google_search(query)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Description: {result['description'][:200]}...")
            print("-" * 80)
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Google Search test completed successfully!")
    else:
        print("\n❌ Google Search test failed!")
        exit(1)
