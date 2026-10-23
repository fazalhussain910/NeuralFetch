import logging
import asyncio
import re
from typing import List, Optional
import httpx
from html.parser import HTMLParser
from app.config import settings
from app.state import ScrapedDocument

logger = logging.getLogger(__name__)

# Fallback HTML text extractor using standard library in case bs4 is not yet installed
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts = []
        self.title_text = ""
        self._in_title = False
        self._ignore_tag = False
        self.ignored_tags = {"script", "style", "nav", "footer", "header", "noscript", "aside", "form"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in self.ignored_tags:
            self._ignore_tag = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() in self.ignored_tags:
            self._ignore_tag = False

    def handle_data(self, data):
        if self._in_title:
            self.title_text += data
        elif not self._ignore_tag:
            clean = data.strip()
            if clean:
                self.text_parts.append(clean)

    def get_text(self) -> str:
        return " ".join(self.text_parts)

class ScraperService:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 NeuralFetch/1.0"
            )
        }

    def search_web(self, query: str, max_results: int = 4) -> List[str]:
        """Free local web search using DDGS with multi-engine fallback to prevent primp / safari_16 errors."""
        urls: List[str] = []
        
        # Method 1: Try modern ddgs package
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for item in results:
                    url = item.get("href") or item.get("link") or item.get("url")
                    if url and url.startswith("http"):
                        urls.append(url)
            if urls:
                logger.info(f"ddgs search for '{query}' returned {len(urls)} URLs.")
                return urls[:max_results]
        except Exception as e:
            logger.debug(f"ddgs module search failed ({e}), attempting duckduckgo_search...")

        # Method 2: Try duckduckgo_search package
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for item in results:
                    url = item.get("href") or item.get("link") or item.get("url")
                    if url and url.startswith("http"):
                        urls.append(url)
            if urls:
                logger.info(f"duckduckgo_search for '{query}' returned {len(urls)} URLs.")
                return urls[:max_results]
        except Exception as e:
            logger.warning(f"duckduckgo_search failed ({e}), using direct HTTP search fallback...")

        # Method 3: Direct HTTP DuckDuckGo HTML endpoint fallback
        try:
            with httpx.Client(timeout=8.0, follow_redirects=True, headers=self.headers) as client:
                resp = client.get("https://html.duckduckgo.com/html/", params={"q": query})
                if resp.status_code == 200 or resp.status_code == 202:
                    # Extract URLs from HTML markup
                    found = re.findall(r'href="//duckduckgo\.com/l/\?uddg=([^"&]+)', resp.text)
                    if not found:
                        found = re.findall(r'<a class="result__url" href="([^"]+)"', resp.text)
                    import urllib.parse
                    for link in found:
                        unquoted = urllib.parse.unquote(link)
                        if unquoted.startswith("http"):
                            urls.append(unquoted)
                    logger.info(f"Direct HTTP search for '{query}' returned {len(urls)} URLs.")
        except Exception as e:
            logger.error(f"Direct HTTP search failed for query '{query}': {e}")

        return urls[:max_results]

    async def scrape_url_fast(self, url: str) -> Optional[ScrapedDocument]:
        """Fast HTTP scraping with BeautifulSoup or standard HTMLParser fallback."""
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=self.headers) as client:
                response = await client.get(url)
                if response.status_code >= 400:
                    logger.warning(f"Failed to fetch {url}: status {response.status_code}")
                    return None

                title = url
                cleaned_text = ""

                # Try BeautifulSoup if available
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else url

                    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
                        element.decompose()

                    raw_text = soup.get_text(separator=" ", strip=True)
                    cleaned_text = " ".join(raw_text.split())
                except ImportError:
                    # Fallback to standard library HTML parser
                    parser = HTMLTextExtractor()
                    parser.feed(response.text)
                    title = parser.title_text.strip() or url
                    cleaned_text = parser.get_text()

                if len(cleaned_text) < 100:
                    logger.info(f"Page content too short ({len(cleaned_text)} chars) for {url}")
                    return None

                return ScrapedDocument(
                    url=url,
                    title=title,
                    content=cleaned_text,
                    length=len(cleaned_text)
                )
        except Exception as e:
            logger.warning(f"Fast scrape failed for {url}: {e}")
            return None

    async def scrape_url_playwright(self, url: str) -> Optional[ScrapedDocument]:
        """Headless Playwright browser scrape for JavaScript-heavy or dynamic SPAs."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=settings.PLAYWRIGHT_HEADLESS)
                page = await browser.new_page(user_agent=self.headers["User-Agent"])
                await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                
                title = await page.title()
                html_content = await page.content()
                await browser.close()

                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, "html.parser")
                    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                        element.decompose()
                    cleaned_text = " ".join(soup.get_text(separator=" ", strip=True).split())
                except ImportError:
                    parser = HTMLTextExtractor()
                    parser.feed(html_content)
                    cleaned_text = parser.get_text()

                return ScrapedDocument(
                    url=url,
                    title=title or url,
                    content=cleaned_text,
                    length=len(cleaned_text)
                )
        except Exception as e:
            logger.error(f"Playwright scrape failed for {url}: {e}")
            return None

    async def scrape_url(self, url: str) -> Optional[ScrapedDocument]:
        """Primary scraper method with fallback to Playwright if dynamic scraping is enabled."""
        doc = await self.scrape_url_fast(url)
        if not doc and settings.USE_DYNAMIC_SCRAPING:
            logger.info(f"Retrying {url} with Playwright engine...")
            doc = await self.scrape_url_playwright(url)
        return doc

    async def scrape_multiple_urls(self, urls: List[str]) -> List[ScrapedDocument]:
        """Scrapes multiple URLs concurrently."""
        tasks = [self.scrape_url(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return [r for r in results if r is not None]

# Singleton instance
scraper_service = ScraperService()
