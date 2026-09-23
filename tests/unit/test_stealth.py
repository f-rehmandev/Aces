import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


async def check(use_stealth: bool):
    if use_stealth:
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            webdriver_flag = await page.evaluate("navigator.webdriver")
            await browser.close()
            return webdriver_flag
    else:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            webdriver_flag = await page.evaluate("navigator.webdriver")
            await browser.close()
            return webdriver_flag


async def main():
    without_stealth = await check(use_stealth=False)
    with_stealth = await check(use_stealth=True)
    print(f"navigator.webdriver WITHOUT stealth: {without_stealth}  (True = detectable as a bot)")
    print(f"navigator.webdriver WITH stealth:    {with_stealth}  (should be False/undefined)")


asyncio.run(main())