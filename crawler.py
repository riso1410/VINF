import os
import re
from urllib.parse import urljoin, urlparse
from collections import deque
import logging
import time
import pickle
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

class Crawler:
    def __init__(self, start_url="https://foodnetwork.co.uk/"):
        # Create directories first
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/raw_html', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize crawler state
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.urls_to_visit = deque([start_url])  # BFS queue
        self.visited_urls = set()
        self.recipe_urls = set()
        
        # Crawler settings
        self.max_urls = None
        self.driver = None
        
        # Initialize Selenium WebDriver
        self.setup_selenium()
        
        # Load checkpoint if exists
        self.checkpoint_file = 'data/crawler_checkpoint.pkl'
        self.load_checkpoint()
    
    def setup_selenium(self):
        """Initialize Selenium WebDriver with Chrome"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        # Suppress common Chrome errors in headless mode
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--disable-background-networking')
        chrome_options.add_argument('--disable-default-apps')
        chrome_options.add_argument('--disable-sync')
        
        # Disable images and CSS for faster loading
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(30)  # 30 seconds timeout
        
        self.logger.info("Selenium WebDriver initialized successfully")
    
    def cleanup_selenium(self):
        """Clean up Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Selenium WebDriver closed")
            except Exception as e:
                self.logger.warning(f"Error closing WebDriver: {e}")
    
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
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    
    def is_recipe_url(self, url):
        """Check if URL is a recipe URL (/recipes/name)"""
        return '/recipes/' in url and url.count('/') >= 4  # Basic pattern matching
    
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
        
        anchor_pattern = r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(anchor_pattern, clean_content, re.IGNORECASE)
        
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
    
    def crawl_page(self, url):
        """Crawl a single page and extract URLs using Selenium"""
        try:
            self.logger.info(f"Crawling: {url}")
            
            # Load page with Selenium
            self.driver.get(url)
            
            # Wait for page to load and JavaScript to execute
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for JavaScript content to load
            time.sleep(3)
            
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
        except WebDriverException as e:
            self.logger.error(f"Selenium error for {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
    
    def save_recipe_urls(self):
        """Save recipe URLs to urls.txt file"""
        try:
            # Read existing URLs from file to avoid duplicates
            existing_urls = set()
            if os.path.exists('data/urls.txt'):
                with open('data/urls.txt', 'r', encoding='utf-8') as f:
                    existing_urls = set(line.strip() for line in f if line.strip())
            
            # Filter out any broken URLs that might contain JSON fragments
            clean_recipe_urls = set()
            for url in self.recipe_urls:
                # Skip URLs that contain JSON-like characters or are malformed
                if any(char in url for char in ['{', '}', '"', '&quot;', '&gt;', '&lt;']):
                    self.logger.warning(f"Skipping malformed URL: {url}")
                    continue
                
                # Check if URL is properly formatted
                try:
                    parsed = urlparse(url)
                    if parsed.scheme and parsed.netloc and '/recipes/' in parsed.path:
                        clean_recipe_urls.add(url)
                except Exception:
                    self.logger.warning(f"Skipping invalid URL: {url}")
                    continue
            
            # Find new URLs that aren't already in the file
            new_urls = clean_recipe_urls - existing_urls
            
            if new_urls:
                # Append new URLs to the file
                with open('data/urls.txt', 'a', encoding='utf-8') as f:
                    for url in sorted(new_urls):
                        f.write(url + '\n')
                self.logger.info(f"Appended {len(new_urls)} new recipe URLs to data/urls.txt")
            else:
                self.logger.info("No new recipe URLs to append")
            
            # Update the recipe_urls set with clean URLs
            self.recipe_urls = clean_recipe_urls
            
        except Exception as e:
            self.logger.error(f"Error saving URLs: {e}")
    
    def crawl(self):
        """Main crawling method using BFS"""
        self.logger.info("Starting BFS crawl...")
        self.logger.info(f"Starting URL: {self.start_url}")
        
        crawled_count = 0
        
        try:
            while self.urls_to_visit and (self.max_urls is None or crawled_count < self.max_urls):
                # Get next URL from queue (BFS)
                current_url = self.urls_to_visit.popleft()
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                
                # Mark as visited
                self.visited_urls.add(current_url)
                crawled_count += 1
                
                # Crawl the page
                self.crawl_page(current_url)
                
                # Save checkpoint periodically
                if crawled_count % 10 == 0:
                    self.save_checkpoint()
                    self.save_recipe_urls()
                    self.logger.info(f"Progress: {crawled_count} pages crawled, {len(self.recipe_urls)} recipes found")
                
                # Respect rate limiting
                time.sleep(random.uniform(2, 5))
                
        except KeyboardInterrupt:
            self.logger.info("Crawling interrupted by user")
        except Exception as e:
            self.logger.error(f"Crawling error: {e}")
        finally:
            # Clean up Selenium
            self.cleanup_selenium()
            
            # Final save
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
