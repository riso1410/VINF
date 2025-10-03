import os
import re
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import pickle
import random
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import config

class Crawler:
    def __init__(self, start_url: str = config.START_URL):            
        # Create directories first
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.RAW_HTML_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Setup logging
        self.logger = config.setup_logging(config.CRAWLER_LOG)
        
        # Initialize crawler state
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.urls_to_visit = deque([start_url])  # BFS queue
        self.visited_urls = set()
        self.recipe_urls = set()
        
        # Crawler settings from config
        self.driver = None
        self.pages_crawled_since_restart = 0
        self.restart_interval = config.RESTART_INTERVAL
        
        # Initialize Selenium WebDriver
        self.setup_selenium()
        
        # Load checkpoint if exists
        self.checkpoint_file = config.CHECKPOINT_FILE
        self.load_checkpoint()
    
    def setup_selenium(self):
        """Initialize Selenium WebDriver with Chrome"""
        chrome_options = config.get_chrome_options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
        self.driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)
        self.logger.info("Selenium WebDriver initialized successfully")
    
    def cleanup_selenium(self):
        """Clean up Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.logger.info("Selenium WebDriver closed")
            self.driver = None
    
    def kill_chrome_processes(self):
        """Kill any remaining Chrome processes"""
        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], 
                      capture_output=True, check=False)
        subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe'], 
                      capture_output=True, check=False)
        self.logger.debug("Attempted to kill Chrome processes")
    
    def restart_selenium(self):
        """Restart Selenium WebDriver session"""
        self.logger.warning("Restarting Selenium WebDriver session...")
        self.cleanup_selenium()
        self.kill_chrome_processes()
        
        time.sleep(3)  # Wait before restarting
        self.setup_selenium()
        self.logger.info("Selenium WebDriver successfully restarted")
        return True
    
    def is_session_valid(self):
        """Check if the current WebDriver session is valid"""
        if not self.driver:
            return False
        try:
            # Try to get current URL as a simple test
            self.driver.current_url
            return True
        except WebDriverException:
            return False
    
    def load_checkpoint(self):
        """Load crawler state from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.urls_to_visit = deque(checkpoint.get('urls_to_visit', [self.start_url]))
                    self.visited_urls = checkpoint.get('visited_urls', set())
                    self.recipe_urls = checkpoint.get('recipe_urls', set())
                self.logger.info(f"Loaded checkpoint: {len(self.visited_urls)} visited, {len(self.recipe_urls)} recipes found")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save crawler state to checkpoint file"""
        checkpoint = {
            'urls_to_visit': list(self.urls_to_visit),
            'visited_urls': self.visited_urls,
            'recipe_urls': self.recipe_urls
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def is_valid_url(self, url):
        """Check if URL is valid for crawling"""
        parsed = urlparse(url)
        
        # Must be from the same domain
        if parsed.netloc != self.domain:
            return False
            
        # Skip certain file types
        if any(url.lower().endswith(ext) for ext in config.SKIP_EXTENSIONS):
            return False
            
        return True
    
    def is_recipe_url(self, url):
        """Check if URL is a recipe URL (/recipes/name)"""
        return config.RECIPE_URL_PATTERN in url and url.count('/') >= config.RECIPE_URL_MIN_SLASHES
    
    def save_html_content(self, url, html_content):
        """Save HTML content to file"""
        parsed = urlparse(url)

        safe_filename = re.sub(r'[^\w\-_.]', '_', parsed.path.strip('/'))
        if not safe_filename:
            safe_filename = 'index'
        
        domain_prefix = re.sub(r'[^\w\-_.]', '_', parsed.netloc)
        filename = f"{domain_prefix}_{safe_filename}.html"
        
        counter = 1
        original_filename = filename
        while os.path.exists(os.path.join(config.RAW_HTML_DIR, filename)):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        filepath = os.path.join(config.RAW_HTML_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.debug(f"Saved HTML content to {filepath}")
        return filepath
    
    def extract_urls_from_html(self, html_content, base_url):
        """Extract URLs from HTML content without using BeautifulSoup select/find methods"""
        urls = set()
        
        clean_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
        
        matches = re.findall(config.ANCHOR_PATTERN, clean_content, re.IGNORECASE)
        
        for match in matches:
            if any(char in match for char in ['{', '}', '"', '&quot;']):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, match)
            
            # Clean up the URL (remove fragments and query parameters for consistency)
            parsed = urlparse(absolute_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            if self.is_valid_url(clean_url):
                urls.add(clean_url)
        
        return urls
    
    def handle_retry_logic(self, url, retry_count, max_retries, error_msg, is_session_error=False):
        """Handle retry logic for crawl_page errors"""
        if retry_count >= max_retries:
            self.logger.error(f"Max retries exceeded for {url}")
            return False
        
        self.logger.info(f"Retrying {url} (attempt {retry_count + 1}/{max_retries})")
        
        self.logger.warning("Session error detected, restarting WebDriver")
        if not self.restart_selenium():
            self.logger.error(f"Failed to restart WebDriver for {url}")
            return False
        time.sleep(3)
    
        self.crawl_page(url, retry_count + 1, max_retries)
        return True
    
    def crawl_page(self, url, retry_count=0, max_retries=None):
        """Crawl a single page and extract URLs using Selenium with retry logic"""
        if max_retries is None:
            max_retries = config.MAX_RETRIES
        
        try:
            # Check if session is valid, restart if needed
            if not self.is_session_valid():
                self.logger.warning("Invalid session detected, restarting WebDriver...")
                if not self.restart_selenium():
                    self.logger.error(f"Failed to restart WebDriver, skipping URL: {url}")
                    return
            
            self.logger.info(f"Crawling: {url}")
            
            # Load page with Selenium
            self.driver.get(url)
            
            # Wait for page to load and JavaScript to execute
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for JavaScript content to load
            time.sleep(config.SELENIUM_JS_WAIT)
            
            # Get the full HTML after JavaScript execution
            html_content = self.driver.page_source
            self.logger.debug(f"Retrieved HTML via Selenium for: {url}")
            
            # Save HTML content to file
            saved_path = self.save_html_content(url, html_content)
            if saved_path:
                self.logger.info(f"Downloaded and saved: {url} -> {saved_path}")
            
            # Extract URLs from the HTML content
            urls = self.extract_urls_from_html(html_content, url)
            
            if self.is_recipe_url(url):
                self.recipe_urls.add(url)
                self.logger.info(f"Found recipe: {url}")
            
            new_urls = sum(1 for found_url in urls if found_url not in self.visited_urls)
            for found_url in urls:
                if found_url not in self.visited_urls:
                    self.urls_to_visit.append(found_url)
            
            self.logger.info(f"Found {new_urls} new URLs on {url}")
            
        except TimeoutException:
            self.logger.error(f"Selenium timeout for {url}")
            self.handle_retry_logic(url, retry_count, max_retries, "timeout")
            
        except WebDriverException as e:
            self.logger.error(f"Selenium error for {url}: {e}")
            
            # Check for session-related errors and restart if needed
            is_session_error = "invalid session id" in str(e).lower() or "session not found" in str(e).lower()
            self.handle_retry_logic(url, retry_count, max_retries, str(e), is_session_error)
            
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            self.handle_retry_logic(url, retry_count, max_retries, str(e))
    
    def crawl(self):
        """Main crawling method using BFS"""
        self.logger.info(f"Starting BFS crawl from {self.start_url}")
        crawled_count = 0

        while self.urls_to_visit:
            current_url = self.urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
            
            self.visited_urls.add(current_url)
            crawled_count += 1
            
            # Restart browser periodically
            if self.pages_crawled_since_restart >= self.restart_interval:
                self.logger.info(f"Restarting browser after {self.restart_interval} pages.")
                if not self.restart_selenium():
                    self.logger.error("Failed to restart WebDriver, stopping crawl.")
                    break
                self.pages_crawled_since_restart = 0
            
            self.crawl_page(current_url)
            self.pages_crawled_since_restart += 1
            
            # Save progress periodically
            if crawled_count % 10 == 0:
                self.save_checkpoint()
                self.logger.info(f"Progress: {crawled_count} pages crawled, {len(self.recipe_urls)} recipes found.")
            
            time.sleep(random.uniform(2, 5))
        
        self.cleanup_selenium()
        self.save_checkpoint()
        self.logger.info("Crawling completed!")
        self.logger.info(f"Total pages visited: {len(self.visited_urls)}")
        self.logger.info(f"Total recipe URLs found: {len(self.recipe_urls)}")

def main():
    """Main function to run the crawler with Selenium"""
    crawler = Crawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
