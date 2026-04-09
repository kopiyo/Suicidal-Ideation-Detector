"""
Test Facebook scraping directly outside Streamlit.
Run this in your terminal:
    python test_facebook.py
"""
from playwright.sync_api import sync_playwright
import time

URL = "https://www.facebook.com/zuck"

print(f"Testing Facebook scraper on: {URL}")
print("-" * 50)

with sync_playwright() as p:
    print("Step 1 - Launching Chromium...")
    browser = p.chromium.launch(
        headless=True,
        args=["--no-sandbox","--disable-dev-shm-usage"]
    )
    print("Step 2 - Creating browser context...")
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 900},
        locale="en-US",
    )
    page = context.new_page()

    print("Step 3 - Navigating to profile...")
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)
        print(f"         Page title: {page.title()}")
        print(f"         Current URL: {page.url}")
    except Exception as e:
        print(f"         ERROR: {e}")
        browser.close()
        raise SystemExit

    # Check if we hit a login wall
    content = page.content()
    if "log in" in content.lower() or "login" in content.lower():
        print("WARNING - Login wall detected. Facebook is requiring login.")
    else:
        print("         No login wall detected.")

    print("Step 4 - Trying to extract post text...")
    selectors = [
        '[data-ad-preview="message"]',
        '[data-testid="post_message"]',
        '.userContent',
        'div[dir="auto"] span[dir="auto"]',
    ]
    found = []
    for sel in selectors:
        els = page.locator(sel).all()
        for el in els[:5]:
            try:
                txt = el.inner_text(timeout=2000).strip()
                if len(txt) > 10:
                    found.append(txt)
            except Exception:
                pass

    found = list(set(found))
    print(f"         Found {len(found)} post(s)")
    for i, t in enumerate(found[:3]):
        print(f"         [{i+1}] {t[:80]}")

    # Save screenshot for inspection
    page.screenshot(path="facebook_debug.png")
    print("Step 5 - Screenshot saved as facebook_debug.png")
    print("         Open this file to see what the browser actually loaded.")

    browser.close()

print("\nTest complete.")
