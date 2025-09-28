#!/usr/bin/env python3
"""
Food Network UK Recipe Crawler

This module crawls the Food Network UK website to extract recipe collections
and download HTML content for each recipe. It supports checkpoint/resume
functionality to handle interruptions during large crawling operations.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
import os
import uuid
import pickle

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "https://foodnetwork.co.uk/recipes"

def setup_logging():
    """
    Setup logging configuration for the crawler.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/crawler.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_directories():
    """
    Create necessary directories for data storage.
    """
    os.makedirs('data/raw_html', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def save_html_page(url, html_content, recipe_name=None):
    """
    Save HTML content to a file with UUID-based filename.
    
    Args:
        url (str): The URL of the recipe page
        html_content (str): Raw HTML content to save
        recipe_name (str, optional): Recipe name for logging purposes
        
    Returns:
        str: Path to the saved file, or None if saving failed
    """
    try:
        file_uuid = str(uuid.uuid4())
        filename = f"data/raw_html/{file_uuid}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        recipe_info = f" ({recipe_name})" if recipe_name else ""
        logging.info(f"Saved HTML: {url}{recipe_info} -> {filename}")
        
        return filename
    except Exception as e:
        logging.error(f"Error saving HTML for {url}: {e}")
        return None

def save_checkpoint(data, checkpoint_name):
    """
    Save crawler state as a checkpoint file.
    
    Args:
        data (dict): Crawler state data to save
        checkpoint_name (str): Name for the checkpoint file
    """
    try:
        checkpoint_file = f"data/checkpoints/{checkpoint_name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logging.error(f"Error saving checkpoint {checkpoint_name}: {e}")

def load_checkpoint(checkpoint_name):
    """
    Load crawler state from a checkpoint file.
    
    Args:
        checkpoint_name (str): Name of the checkpoint file to load
        
    Returns:
        dict: Loaded checkpoint data, or None if loading failed
    """
    try:
        checkpoint_file = f"data/checkpoints/{checkpoint_name}.pkl"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Checkpoint loaded: {checkpoint_file}")
            return data
        else:
            logging.info(f"No checkpoint found: {checkpoint_file}")
            return None
    except Exception as e:
        logging.error(f"Error loading checkpoint {checkpoint_name}: {e}")
        return None

def get_latest_checkpoint():
    """
    Find the most recent checkpoint file by modification time.
    
    Returns:
        str: Name of the latest checkpoint (without .pkl extension), or None if none found
    """
    try:
        checkpoint_dir = "data/checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        if not checkpoint_files:
            return None
        
        # Sort by modification time, get the latest
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        latest_file = checkpoint_files[0]
        return latest_file[:-4]  # Remove .pkl extension
    except Exception as e:
        logging.error(f"Error getting latest checkpoint: {e}")
        return None

def collect_collection_urls(base_url=BASE_URL):
    """
    Collect all recipe collection URLs from the Food Network UK recipes page.
    
    Args:
        base_url (str): The base URL to scrape for collection links
        
    Returns:
        list: Sorted list of collection URLs found on the page
    """
    try:
        logging.info(f"Fetching collection URLs from: {base_url}")
        
        response = requests.get(base_url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        collection_urls = set()
        
        for link in links:
            href = link['href']
            
            # Convert relative URLs to absolute URLs
            if href.startswith('/'):
                full_url = urljoin(base_url, href)
            elif href.startswith('http'):
                full_url = href
            else:
                continue
            
            # Filter for collection URLs
            if '/collections/' in full_url and 'foodnetwork.co.uk' in full_url:
                # Remove query parameters to get clean collection URL
                parsed = urlparse(full_url)
                base_collection_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                collection_urls.add(base_collection_url)
        
        logging.info(f"Found {len(collection_urls)} collection URLs")
        return sorted(list(collection_urls))
        
    except requests.RequestException as e:
        logging.error(f"Error fetching the page: {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing the page: {e}")
        return []

def extract_recipe_urls_from_page(soup, base_url):
    """
    Extract individual recipe URLs from a collection page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content of the collection page
        base_url (str): Base URL for resolving relative links
        
    Returns:
        set: Set of recipe URLs found on the page
    """
    recipe_urls = set()
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link['href']
        
        # Convert relative URLs to absolute URLs
        if href.startswith('/'):
            full_url = urljoin(base_url, href)
        elif href.startswith('http'):
            full_url = href
        else:
            continue
        
        # Filter for recipe URLs (exclude collection URLs)
        if '/recipes/' in full_url and 'foodnetwork.co.uk' in full_url and '/collections/' not in full_url:
            recipe_urls.add(full_url)
    
    return recipe_urls

def get_pagination_urls(soup, base_collection_url):
    """
    Extract pagination URLs from a collection page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content of the collection page
        base_collection_url (str): Base URL of the current collection
        
    Returns:
        set: Set of pagination URLs for the same collection
    """
    pagination_urls = set()
    
    # Look for pagination container
    pagination_div = soup.find('div', class_='pagination-items')
    if pagination_div:
        links = pagination_div.find_all('a', href=True)
        collection_name = base_collection_url.split('/')[-1]
        
        for link in links:
            href = link['href']
            
            # Convert relative URLs to absolute URLs
            if href.startswith('/'):
                full_url = urljoin(base_collection_url, href)
            elif href.startswith('http'):
                full_url = href
            else:
                continue
            
            # Only include pagination URLs for the same collection
            if collection_name in full_url:
                pagination_urls.add(full_url)
    
    return pagination_urls

def crawl_collection_recipes(collection_url):
    """
    Crawl all recipes from a collection, handling pagination automatically.
    
    Args:
        collection_url (str): URL of the collection to crawl
        
    Returns:
        set: Set of all recipe URLs found in the collection
    """
    all_recipe_urls = set()
    visited_pages = set()
    pages_to_visit = {collection_url}
    page_count = 0
    
    collection_name = collection_url.split('/')[-1]
    logging.info(f"Starting to crawl collection: {collection_name}")
    
    while pages_to_visit:
        current_page = pages_to_visit.pop()
        
        if current_page in visited_pages:
            continue
            
        visited_pages.add(current_page)
        page_count += 1
        
        try:
            logging.debug(f"Fetching page {page_count}: {current_page}")
            response = requests.get(current_page, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract recipe URLs from current page
            page_recipe_urls = extract_recipe_urls_from_page(soup, collection_url)
            all_recipe_urls.update(page_recipe_urls)
            logging.debug(f"Found {len(page_recipe_urls)} recipes on page {page_count}")
            
            # Get pagination URLs and add unvisited ones to queue
            pagination_urls = get_pagination_urls(soup, collection_url)
            for pag_url in pagination_urls:
                if pag_url not in visited_pages:
                    pages_to_visit.add(pag_url)
            
            time.sleep(1)  # Rate limiting
            
        except requests.RequestException as e:
            logging.error(f"Error fetching page {current_page}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing page {current_page}: {e}")
            continue
    
    logging.info(f"Collection {collection_name}: {len(all_recipe_urls)} recipes found across {page_count} pages")
    return all_recipe_urls

def download_recipe_html(recipe_url, recipe_name=None):
    """
    Download and save HTML content of a recipe page.
    
    Args:
        recipe_url (str): URL of the recipe page to download
        recipe_name (str, optional): Recipe name for logging purposes
        
    Returns:
        str: Path to the saved HTML file, or None if download failed
    """
    try:
        response = requests.get(recipe_url, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        
        return save_html_page(recipe_url, response.text, recipe_name)
        
    except requests.RequestException as e:
        logging.error(f"Error downloading recipe {recipe_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing recipe {recipe_url}: {e}")
        return None

def crawl_all_collections(collection_urls, download_html=True, resume_from_checkpoint=True):
    """
    Crawl all collections and extract recipe URLs with optional HTML download.
    Supports checkpoint/resume functionality for interrupted operations.
    
    Args:
        collection_urls (list): List of collection URLs to crawl
        download_html (bool): Whether to download HTML content for each recipe
        resume_from_checkpoint (bool): Whether to resume from existing checkpoint
        
    Returns:
        tuple: (all_recipes dict, total_recipes count)
    """
    # Initialize or resume from checkpoint
    checkpoint_data = None
    if resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_data = load_checkpoint(latest_checkpoint)
    
    if checkpoint_data:
        all_recipes = checkpoint_data.get('all_recipes', {})
        total_recipes = checkpoint_data.get('total_recipes', 0)
        processed_collections = checkpoint_data.get('processed_collections', set())
        logging.info(f"Resuming from checkpoint: {len(processed_collections)} collections already processed")
    else:
        all_recipes = {}
        total_recipes = 0
        processed_collections = set()
    
    collections_processed_this_session = 0
    
    for i, collection_url in enumerate(collection_urls):
        collection_name = collection_url.split('/')[-1]
        
        # Skip already processed collections
        if collection_name in processed_collections:
            logging.info(f"Skipping already processed collection: {collection_name}")
            continue
            
        logging.info(f"Processing collection {i+1}/{len(collection_urls)}: {collection_name}")
        
        try:
            recipe_urls = crawl_collection_recipes(collection_url)
            recipes_data = []
            
            if download_html and recipe_urls:
                logging.info(f"Downloading HTML for {len(recipe_urls)} recipes in {collection_name}")
                
                for recipe_url in recipe_urls:
                    # Extract recipe name from URL for logging
                    recipe_name = recipe_url.split('/')[-1] if recipe_url else "unknown"
                    html_filename = download_recipe_html(recipe_url, recipe_name)
                    
                    recipes_data.append({
                        'url': recipe_url,
                        'name': recipe_name,
                        'html_file': html_filename
                    })
                    
                    time.sleep(1)  # Rate limiting
            else:
                # Store URLs without downloading HTML
                for recipe_url in recipe_urls:
                    recipe_name = recipe_url.split('/')[-1] if recipe_url else "unknown"
                    recipes_data.append({
                        'url': recipe_url,
                        'name': recipe_name,
                        'html_file': None
                    })
            
            # Update collection data
            all_recipes[collection_name] = {
                'url': collection_url,
                'recipe_count': len(recipes_data),
                'recipes': recipes_data
            }
            total_recipes += len(recipes_data)
            processed_collections.add(collection_name)
            collections_processed_this_session += 1
            
            logging.info(f"Completed collection {collection_name}: {len(recipes_data)} recipes")

            # Save checkpoint after each collection
            checkpoint_data = {
                'all_recipes': all_recipes,
                'total_recipes': total_recipes,
                'processed_collections': processed_collections,
                'collection_urls': collection_urls,
                'timestamp': time.time()
            }
            checkpoint_name = f"crawler_checkpoint_{int(time.time())}"
            save_checkpoint(checkpoint_data, checkpoint_name)
                
        except Exception as e:
            logging.error(f"Error processing collection {collection_name}: {e}")
            all_recipes[collection_name] = {
                'url': collection_url,
                'recipe_count': 0,
                'recipes': [],
                'error': str(e)
            }
    
    return all_recipes, total_recipes

def main():
    """Main function to execute the crawling process."""
    setup_directories()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Food Network UK Recipe Crawler Started")
    logger.info("=" * 60)
    
    try:
        # Load collection URLs from checkpoint or fetch new ones
        collection_urls = None
        latest_checkpoint = get_latest_checkpoint()
        
        if latest_checkpoint:
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            checkpoint_data = load_checkpoint(latest_checkpoint)
            if checkpoint_data and 'collection_urls' in checkpoint_data:
                collection_urls = checkpoint_data['collection_urls']
                logger.info(f"Loaded {len(collection_urls)} collection URLs from checkpoint")
            
        if not collection_urls:
            logger.info("[STEP 1] Collecting collection URLs from main recipes page...")
            collection_urls = collect_collection_urls()
            
            if not collection_urls:
                logger.error("No collection URLs found or error occurred.")
                return
        
        logger.info(f"Found {len(collection_urls)} collection URLs")
        for i, url in enumerate(collection_urls, 1):
            collection_name = url.split('/')[-1]
            logger.debug(f"  {i:2d}. {collection_name}")
                
        # Crawl all collections and download HTML
        logger.info(f"[STEP 2] Crawling {len(collection_urls)} collections for recipes and downloading HTML...")
        all_recipes, total_recipes = crawl_all_collections(
            collection_urls, 
            download_html=True, 
            resume_from_checkpoint=True
        )
        
        # Display summary statistics
        logger.info("=" * 60)
        logger.info("CRAWLING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total collections processed: {len(collection_urls)}")
        logger.info(f"Total recipes found: {total_recipes}")
        
        if collection_urls:
            logger.info(f"Average recipes per collection: {total_recipes / len(collection_urls):.1f}")
        
        # Count successfully saved HTML files
        html_files_count = sum(
            1 for collection_data in all_recipes.values() 
            for recipe_data in collection_data['recipes'] 
            if isinstance(recipe_data, dict) and recipe_data.get('html_file')
        )
        logger.info(f"Total HTML files saved: {html_files_count}")
        
        # Show top collections by recipe count
        sorted_collections = sorted(all_recipes.items(), key=lambda x: x[1]['recipe_count'], reverse=True)
        logger.info("Top 10 collections by recipe count:")
        for i, (name, data) in enumerate(sorted_collections[:10], 1):
            logger.info(f"  {i:2d}. {name}: {data['recipe_count']} recipes")
        
        logger.info("Crawling completed successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()