"""
Web crawler for websites using Selenium WebDriver.
Performs breadth-first search crawling and saves all HTML pages.
"""

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
    """
    Web crawler that performs breadth-first search of a website.
    Uses Selenium WebDriver for JavaScript-rendered content.
    """
    
    def __init__(self, start_url: str = config.START_URL):
        """
        Initialize the crawler with necessary directories and state.
        
        Args:
            start_url: URL to start crawling from
        """
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.RAW_HTML_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.CRAWLER_LOG)
        
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.urls_to_visit = deque([start_url])
        self.visited_urls = set()
        self.downloaded_recipes = set()
        self.recipes_saved = 0  
        
        self.driver = None
        self.pages_crawled_since_restart = 0
        self.restart_interval = config.RESTART_INTERVAL
        
        self.setup_selenium()
        
        self.checkpoint_file = config.CHECKPOINT_FILE
        self.load_checkpoint()
    
    def setup_selenium(self):
        """Initialize Selenium WebDriver with Chrome options."""
        chrome_options = config.get_chrome_options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
        self.driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)
    
    def cleanup_selenium(self):
        """Clean up and quit the Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def kill_chrome_processes(self):
        """Forcefully terminate any remaining Chrome processes."""
        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], 
                      capture_output=True, check=False)
        subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe'], 
                      capture_output=True, check=False)
    
    def restart_selenium(self):
        """
        Restart the Selenium WebDriver session.
        
        Returns:
            bool: True if restart successful
        """
        self.cleanup_selenium()
        self.kill_chrome_processes()
        
        time.sleep(3)
        self.setup_selenium()
        return True
    
    def is_session_valid(self):
        """
        Check if the current WebDriver session is valid.
        
        Returns:
            bool: True if session is valid
        """
        if not self.driver:
            return False
        try:
            self.driver.current_url
            return True
        except WebDriverException:
            return False
    
    def load_checkpoint(self):
        """Load crawler state from checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.urls_to_visit = deque(checkpoint.get('urls_to_visit', [self.start_url]))
                    self.visited_urls = checkpoint.get('visited_urls', set())
                    self.downloaded_recipes = checkpoint.get('downloaded_recipes', set())
                    self.recipes_saved = len(self.downloaded_recipes)
                self.logger.info(f"Loaded checkpoint: {len(self.visited_urls)} pages visited, {self.recipes_saved} recipes downloaded")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current crawler state to checkpoint file."""
        checkpoint = {
            'urls_to_visit': list(self.urls_to_visit),
            'visited_urls': self.visited_urls,
            'downloaded_recipes': self.downloaded_recipes
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def save_html_content(self, url, html_content):
        """
        Save HTML content to file with unique filename.
        
        Args:
            url: Source URL
            html_content: HTML content to save
            
        Returns:
            str: File path where content was saved
        """
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
        
        return filepath
    
    def is_recipe_url(self, url):
        """
        Check if URL is a recipe page based on URL pattern.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL matches recipe pattern
        """
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]

        if len(path_parts) != 2:
            return False
        
        if path_parts[0] != 'recipes':
            return False
        
        return True
    
    def extract_urls_from_html(self, html_content, base_url):
        """
        Extract URLs from HTML content using regex patterns.
        Extracts URLs to continue crawling, but only recipe URLs will be saved.
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative URLs
            
        Returns:
            set: Set of absolute URLs found in content
        """
        urls = set()
        
        clean_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
        
        matches = re.findall(config.ANCHOR_PATTERN, clean_content, re.IGNORECASE)
        
        for match in matches:
            if any(char in match for char in ['{', '}', '"', '&quot;']):
                continue
            
            absolute_url = urljoin(base_url, match)
            
            clean_url = absolute_url.split('#')[0]
            
            parsed = urlparse(clean_url)
            
            if parsed.netloc != self.domain:
                continue
            
            if any(clean_url.lower().endswith(ext) for ext in config.SKIP_EXTENSIONS):
                continue
            
            urls.add(clean_url)
        
        return urls
    
    def handle_retry_logic(self, url, retry_count, max_retries, error_msg, is_session_error=False):
        """
        Handle retry logic for failed page crawls.
        
        Args:
            url: URL that failed
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            error_msg: Error message from the failure
            is_session_error: Whether the error is session-related
            
        Returns:
            bool: True if retry was successful
        """
        if retry_count >= max_retries:
            self.logger.error(f"Max retries exceeded for {url}")
            return False
        
        self.logger.info(f"Retrying {url} (attempt {retry_count + 1}/{max_retries})")
        
        if not self.restart_selenium():
            self.logger.error(f"Failed to restart WebDriver for {url}")
            return False
        time.sleep(3)
    
        self.crawl_page(url, retry_count + 1, max_retries)
        return True
    
    def crawl_page(self, url, retry_count=0, max_retries=None):
        """
        Crawl a single page and extract URLs using Selenium.
        
        Args:
            url: URL to crawl
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
        """
        if max_retries is None:
            max_retries = config.MAX_RETRIES
        
        try:
            if not self.is_session_valid():
                if not self.restart_selenium():
                    self.logger.error(f"Failed to restart WebDriver, skipping URL: {url}")
                    return
            
            self.logger.info(f"Crawling: {url}")
            
            self.driver.get(url)
            
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(config.SELENIUM_JS_WAIT)
            
            html_content = self.driver.page_source
            
            if self.is_recipe_url(url):
                if url not in self.downloaded_recipes:
                    saved_path = self.save_html_content(url, html_content)
                    if saved_path:
                        self.downloaded_recipes.add(url)
                        self.recipes_saved += 1
                        self.logger.info(f"Saved recipe {self.recipes_saved}: {url}")
            
            urls = self.extract_urls_from_html(html_content, url)
            
            new_urls = sum(1 for found_url in urls if found_url not in self.visited_urls)
            new_recipe_urls = sum(1 for found_url in urls if found_url not in self.visited_urls and self.is_recipe_url(found_url))
            
            for found_url in urls:
                if found_url not in self.visited_urls:
                    self.urls_to_visit.append(found_url)
            
            if new_recipe_urls > 0:
                self.logger.info(f"Found {new_urls} new URLs ({new_recipe_urls} recipes) on {url}")
            else:
                self.logger.info(f"Found {new_urls} new URLs (0 recipes) on {url}")
            
        except TimeoutException:
            self.logger.error(f"Selenium timeout for {url}")
            self.handle_retry_logic(url, retry_count, max_retries, "timeout")
            
        except WebDriverException as e:
            self.logger.error(f"Selenium error for {url}: {e}")
            
            is_session_error = "invalid session id" in str(e).lower() or "session not found" in str(e).lower()
            self.handle_retry_logic(url, retry_count, max_retries, str(e), is_session_error)
            
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            self.handle_retry_logic(url, retry_count, max_retries, str(e))
    
    def crawl(self):
        """Main crawling method using breadth-first search."""
        self.logger.info(f"Starting BFS crawl from {self.start_url}")
        crawled_count = 0

        while self.urls_to_visit:
            current_url = self.urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
            
            self.visited_urls.add(current_url)
            crawled_count += 1
            
            if self.pages_crawled_since_restart >= self.restart_interval:
                if not self.restart_selenium():
                    self.logger.error("Failed to restart WebDriver, stopping crawl.")
                    break
                self.pages_crawled_since_restart = 0
            
            self.crawl_page(current_url)
            self.pages_crawled_since_restart += 1
            
            if crawled_count % 10 == 0:
                self.save_checkpoint()
                self.logger.info(f"Progress: {crawled_count} pages crawled, {self.recipes_saved} recipes saved")
            
            time.sleep(random.uniform(2, 5))
        
        self.cleanup_selenium()
        self.save_checkpoint()
        self.logger.info("Crawling completed!")
        self.logger.info(f"Total pages visited: {len(self.visited_urls)}")
        self.logger.info(f"Total recipes saved: {self.recipes_saved}")


def main():
    """Main function to run the crawler."""
    crawler = Crawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
