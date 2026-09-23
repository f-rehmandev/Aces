import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


async def main():
    async with Stealth().use_async(async_playwright()) as p:
        browser = await p.chromium.launch(headless=False)  # visible window this time
        page = await browser.new_page()
        print("Navigating... watch the browser window that pops up.")
        try:
            await page.goto("https://www.daraz.pk/mouse/", wait_until="domcontentloaded", timeout=60000)
            print("SUCCESS — page loaded.")
        except Exception as e:
            print(f"FAILED: {e}")
        print("\nLeaving browser open for 15 seconds so you can look at it...")
        await asyncio.sleep(15)
        await browser.close()


asyncio.run(main())