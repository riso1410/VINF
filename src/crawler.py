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
        self.max_urls = config.MAX_URLS
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
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
            self.driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def cleanup_selenium(self):
        """Clean up Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Selenium WebDriver closed")
            except Exception as e:
                self.logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None
    
    def kill_chrome_processes(self):
        """Kill any remaining Chrome processes"""
        try:
            # Kill Chrome processes on Windows
            subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], 
                          capture_output=True, check=False)
            subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe'], 
                          capture_output=True, check=False)
            self.logger.debug("Attempted to kill Chrome processes")
        except Exception as e:
            self.logger.debug(f"Could not kill Chrome processes: {e}")
    
    def restart_selenium(self):
        """Restart Selenium WebDriver session"""
        self.logger.warning("Restarting Selenium WebDriver session...")
        self.cleanup_selenium()
        
        # Kill any remaining Chrome processes
        self.kill_chrome_processes()
        
        time.sleep(3)  # Wait before restarting
        try:
            self.setup_selenium()
            self.logger.info("Selenium WebDriver successfully restarted")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart WebDriver: {e}")
            return False
    
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
        try:
            checkpoint = {
                'urls_to_visit': list(self.urls_to_visit),
                'visited_urls': self.visited_urls,
                'recipe_urls': self.recipe_urls
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint: {e}")
    
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
        try:
            # Create a safe filename from the URL
            parsed = urlparse(url)
            # Replace special characters and create a unique filename
            safe_filename = re.sub(r'[^\w\-_.]', '_', parsed.path.strip('/'))
            if not safe_filename:
                safe_filename = 'index'
            
            # Add domain prefix to avoid conflicts
            domain_prefix = re.sub(r'[^\w\-_.]', '_', parsed.netloc)
            filename = f"{domain_prefix}_{safe_filename}.html"
            
            # Ensure unique filename if it already exists
            counter = 1
            original_filename = filename
            while os.path.exists(f"data/raw_html/{filename}"):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1
            
            filepath = f"data/raw_html/{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.debug(f"Saved HTML content to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving HTML content for {url}: {e}")
            return None
    
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
            
            # Check if current URL is a recipe URL
            if self.is_recipe_url(url):
                self.recipe_urls.add(url)
                self.logger.info(f"Found recipe: {url}")
            
            # Add new URLs to the queue (BFS)
            new_urls = 0
            for found_url in urls:
                if found_url not in self.visited_urls:
                    self.urls_to_visit.append(found_url)
                    new_urls += 1
            
            self.logger.info(f"Found {new_urls} new URLs on {url}")
            
        except TimeoutException:
            self.logger.error(f"Selenium timeout for {url}")
            if retry_count < max_retries:
                self.logger.info(f"Retrying {url} (attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)  # Wait before retry
                self.crawl_page(url, retry_count + 1, max_retries)
            
        except WebDriverException as e:
            self.logger.error(f"Selenium error for {url}: {e}")
            
            # Check for session-related errors and restart if needed
            if "invalid session id" in str(e).lower() or "session not found" in str(e).lower():
                if retry_count < max_retries:
                    self.logger.warning(f"Session error detected, restarting and retrying {url}")
                    if self.restart_selenium():
                        time.sleep(3)
                        self.crawl_page(url, retry_count + 1, max_retries)
                    else:
                        self.logger.error(f"Failed to restart WebDriver for {url}")
                else:
                    self.logger.error(f"Max retries exceeded for {url}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            if retry_count < max_retries:
                self.logger.info(f"Retrying {url} due to unexpected error (attempt {retry_count + 1}/{max_retries})")
                time.sleep(3)
                self.crawl_page(url, retry_count + 1, max_retries)
    
    def save_recipe_urls(self):
        """Save recipe URLs to urls.txt file"""
        try:
            existing_urls = set()
            if os.path.exists(config.URLS_FILE):
                with open(config.URLS_FILE, 'r', encoding='utf-8') as f:
                    existing_urls = set(line.strip() for line in f if line.strip())

            clean_recipe_urls = set()
            for url in self.recipe_urls:
                if any(char in url for char in ['{', '}', '"', '&quot;', '&gt;', '&lt;']):
                    self.logger.warning(f"Skipping malformed URL: {url}")
                    continue
                
                parsed = urlparse(url)
                if not (parsed.scheme and parsed.netloc and config.RECIPE_URL_PATTERN in parsed.path):
                    self.logger.warning(f"Skipping invalid URL format: {url}")
                    continue
                clean_recipe_urls.add(url)

            new_urls = clean_recipe_urls - existing_urls
            
            if new_urls:
                with open(config.URLS_FILE, 'a', encoding='utf-8') as f:
                    for url in sorted(new_urls):
                        f.write(url + '\n')
                self.logger.info(f"Appended {len(new_urls)} new recipe URLs to {config.URLS_FILE}")
            else:
                self.logger.info("No new recipe URLs to append")
            
            self.recipe_urls = clean_recipe_urls
            
        except Exception as e:
            self.logger.error(f"Error saving URLs: {e}")

    def crawl(self):
        """Main crawling method using BFS"""
        self.logger.info(f"Starting BFS crawl from {self.start_url}")
        crawled_count = 0
        
        try:
            while self.urls_to_visit and (self.max_urls is None or crawled_count < self.max_urls):
                current_url = self.urls_to_visit.popleft()
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                crawled_count += 1
                
                if self.pages_crawled_since_restart >= self.restart_interval:
                    self.logger.info(f"Restarting browser after {self.restart_interval} pages.")
                    if not self.restart_selenium():
                        self.logger.error("Failed to restart WebDriver, stopping crawl.")
                        break
                    self.pages_crawled_since_restart = 0
                
                self.crawl_page(current_url)
                self.pages_crawled_since_restart += 1
                
                if crawled_count % 10 == 0:
                    self.save_checkpoint()
                    self.save_recipe_urls()
                    self.logger.info(f"Progress: {crawled_count} pages crawled, {len(self.recipe_urls)} recipes found.")
                
                time.sleep(random.uniform(2, 5))
                
        except KeyboardInterrupt:
            self.logger.info("Crawling interrupted by user.")
        except Exception as e:
            self.logger.critical(f"A critical error occurred during crawling: {e}", exc_info=True)
        finally:
            self.logger.info("Crawling finished or was interrupted. Cleaning up...")
            self.cleanup_selenium()
            self.save_checkpoint()
            self.save_recipe_urls()
            self.logger.info("Crawling completed!")
            self.logger.info(f"Total pages visited: {len(self.visited_urls)}")
            self.logger.info(f"Total recipe URLs found: {len(self.recipe_urls)}")

def main():
    """Main function to run the crawler with Selenium"""
    crawler = Crawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
