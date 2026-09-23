import asyncio
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from scraper.engine import ScraperEngine

async def main():
    scraper = ScraperEngine()
    html = await scraper.fetch_html("https://www.yellowpages.com/search?search_terms=pizza+shops")
    print(html[:2000])

asyncio.run(main())