import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
import os
import uuid

def setup_logging():
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
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
    Create necessary directories
    """
    os.makedirs('data/raw_html', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def save_html_page(url, html_content, recipe_name=None):
    """
    Save HTML content to file with UUID filename
    """
    try:
        # Generate UUID for filename
        file_uuid = str(uuid.uuid4())
        filename = f"data/raw_html/{file_uuid}.html"
        
        # Save HTML content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Log the save operation
        recipe_info = f" ({recipe_name})" if recipe_name else ""
        logging.info(f"Saved HTML: {url}{recipe_info} -> {filename}")
        
        return filename
    except Exception as e:
        logging.error(f"Error saving HTML for {url}: {e}")
        return None

def collect_collection_urls(base_url="https://foodnetwork.co.uk/recipes"):
    """
    Collect all collection/* URLs from the Food Network UK recipes page
    """
    try:
        logging.info(f"Fetching collection URLs from: {base_url}")
        
        # Send GET request to the page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        # Extract collection URLs
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
            
            # Check if the URL contains '/collections/'
            if '/collections/' in full_url and 'foodnetwork.co.uk' in full_url:
                # Remove query parameters to get base collection URL
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
    Extract recipe URLs from a collection page
    """
    recipe_urls = set()
    
    # Find all links that point to recipes
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
        
        # Check if the URL is a recipe URL (contains '/recipes/' and recipe name)
        if '/recipes/' in full_url and 'foodnetwork.co.uk' in full_url:
            # Make sure it's not a collection URL
            if '/collections/' not in full_url:
                recipe_urls.add(full_url)
    
    return recipe_urls

def get_pagination_urls(soup, base_collection_url):
    """
    Extract pagination URLs from the collection page
    """
    pagination_urls = set()
    
    # Look for pagination links
    pagination_div = soup.find('div', class_='pagination-items')
    if pagination_div:
        links = pagination_div.find_all('a', href=True)
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
            if base_collection_url.split('/')[-1] in full_url:
                pagination_urls.add(full_url)
    
    return pagination_urls

def crawl_collection_recipes(collection_url):
    """
    Crawl all recipes from a collection, handling pagination
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
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
            response = requests.get(current_page, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract recipe URLs from current page
            page_recipe_urls = extract_recipe_urls_from_page(soup, collection_url)
            all_recipe_urls.update(page_recipe_urls)
            logging.debug(f"Found {len(page_recipe_urls)} recipes on page {page_count}")
            
            # Get pagination URLs
            pagination_urls = get_pagination_urls(soup, collection_url)
            
            # Add new pagination URLs to visit
            for pag_url in pagination_urls:
                if pag_url not in visited_pages:
                    pages_to_visit.add(pag_url)
            
            # Be respectful to the server
            time.sleep(1)
            
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
    Download HTML content of a recipe page
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(recipe_url, headers=headers)
        response.raise_for_status()
        
        # Save the HTML content
        filename = save_html_page(recipe_url, response.text, recipe_name)
        return filename
        
    except requests.RequestException as e:
        logging.error(f"Error downloading recipe {recipe_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing recipe {recipe_url}: {e}")
        return None

def crawl_all_collections(collection_urls, download_html=True):
    """
    Crawl all collections and extract recipe URLs, optionally download HTML
    """
    all_recipes = {}
    total_recipes = 0
    
    for collection_url in collection_urls:
        collection_name = collection_url.split('/')[-1]
        logging.info(f"Processing collection: {collection_name}")
        
        try:
            recipe_urls = crawl_collection_recipes(collection_url)
            
            # Prepare recipe data
            recipes_data = []
            
            if download_html and recipe_urls:
                logging.info(f"Downloading HTML for {len(recipe_urls)} recipes in {collection_name}")
                
                for recipe_url in recipe_urls:
                    # Extract recipe name from URL for better logging
                    recipe_name = recipe_url.split('/')[-1] if recipe_url else "unknown"
                    
                    # Download HTML
                    html_filename = download_recipe_html(recipe_url, recipe_name)
                    
                    recipes_data.append({
                        'url': recipe_url,
                        'name': recipe_name,
                        'html_file': html_filename
                    })
                    
                    # Be respectful to the server
                    time.sleep(1)
                else:
                    # Just store URLs without downloading HTML
                    for recipe_url in recipe_urls:
                        recipe_name = recipe_url.split('/')[-1] if recipe_url else "unknown"
                        recipes_data.append({
                            'url': recipe_url,
                            'name': recipe_name,
                            'html_file': None
                        })
                
                all_recipes[collection_name] = {
                    'url': collection_url,
                    'recipe_count': len(recipes_data),
                    'recipes': recipes_data
                }
                total_recipes += len(recipes_data)
                
                logging.info(f"Completed collection {collection_name}: {len(recipes_data)} recipes")
                
        except Exception as e:
            logging.error(f"Error processing collection {collection_name}: {e}")
            all_recipes[collection_name] = {
                'url': collection_url,
                'recipe_count': 0,
                'recipes': [],
                'error': str(e)
            }
    
    return all_recipes, total_recipes

if __name__ == "__main__":
    # Setup logging and directories
    setup_directories()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Food Network UK Recipe Crawler Started")
    logger.info("=" * 60)
    
    try:
        # Step 1: Collect collection URLs
        logger.info("[STEP 1] Collecting collection URLs from main recipes page...")
        collection_urls = collect_collection_urls()
        
        if not collection_urls:
            logger.error("No collection URLs found or error occurred.")
            exit(1)
        
        logger.info(f"Found {len(collection_urls)} collection URLs")
        for i, url in enumerate(collection_urls, 1):
            collection_name = url.split('/')[-1]
            logger.debug(f"  {i:2d}. {collection_name}")
                
        # Step 2: Crawl all collections for recipes and download HTML
        logger.info(f"[STEP 2] Crawling {len(collection_urls)} collections for recipes and downloading HTML...")
        all_recipes, total_recipes = crawl_all_collections(collection_urls, download_html=True)
        
        # Step 3: Results logged only (no file saving)
        
        # Summary
        logger.info("=" * 60)
        logger.info("CRAWLING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total collections processed: {len(collection_urls)}")
        logger.info(f"Total recipes found: {total_recipes}")
        
        if len(collection_urls) > 0:
            logger.info(f"Average recipes per collection: {total_recipes / len(collection_urls):.1f}")
        
        # Count HTML files saved
        html_files_count = sum(
            1 for collection_data in all_recipes.values() 
            for recipe_data in collection_data['recipes'] 
            if isinstance(recipe_data, dict) and recipe_data.get('html_file')
        )
        logger.info(f"Total HTML files saved: {html_files_count}")
        
        logger.info("Output files:")
        logger.info(f"- data/raw_html/ (directory with {html_files_count} HTML files)")
        logger.info("- All other results logged only (no JSON/TXT files saved)")
        
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