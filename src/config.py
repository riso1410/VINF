"""
Configuration file for the Recipe Search Engine
Contains all hyperparameters and settings for crawler, scraper, indexer, and search components.
"""

import logging
import os

from selenium.webdriver.chrome.options import Options

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_HTML_DIR = os.path.join(PROJECT_ROOT, "data", "raw_html")
SCRAPED_DIR = os.path.join(PROJECT_ROOT, "data", "scraped")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "index")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# File paths
URLS_FILE = os.path.join(PROJECT_ROOT, "data", "urls.txt")
RECIPES_FILE = os.path.join(PROJECT_ROOT, "data", "scraped", "recipes.jsonl")
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "data", "crawler_checkpoint.pkl")
INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "index", "search_index.pkl")

# Log files
CRAWLER_LOG = os.path.join(PROJECT_ROOT, "logs", "crawler.log")
SCRAPER_LOG = os.path.join(PROJECT_ROOT, "logs", "scraper.log")
INDEXER_LOG = os.path.join(PROJECT_ROOT, "logs", "indexer.log")
SEARCH_LOG = os.path.join(PROJECT_ROOT, "logs", "search.log")

# ============================================================================
# CRAWLER SETTINGS
# ============================================================================

# Starting URL for crawling
START_URL = "https://foodnetwork.co.uk/"

# Crawler behavior
RESTART_INTERVAL = 50  # Restart browser every N pages to prevent memory issues

# Selenium/Chrome settings
SELENIUM_PAGE_LOAD_TIMEOUT = 30  # seconds
SELENIUM_IMPLICIT_WAIT = 10  # seconds
SELENIUM_JS_WAIT = 2  # Additional wait for JavaScript to load (seconds)

# URL validation
SKIP_EXTENSIONS = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".xml"]

# Chrome options
CHROME_WINDOW_SIZE = "1920,1080"
CHROME_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (FIIT School Project - VINF Course)"
# Retry settings
MAX_RETRIES = 3

# ============================================================================
# INDEXER SETTINGS
# ============================================================================

# Text processing
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 25

# Progress logging
INDEXER_PROGRESS_INTERVAL = 200  # Log progress every N documents

# Stop words for indexing and search
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "or",
    "but",
    "if",
    "then",
    "this",
    "can",
    "add",
    "use",
    "place",
    "put",
    "serve",
    "until",
    "about",
    "over",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "now",
    "recipe",
    "recipes",
}

# ============================================================================
# SEARCH SETTINGS
# ============================================================================

# Search behavior
DEFAULT_TOP_K = 5  # Default number of search results to return

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============================================================================
# SPARK EXTRACTION SETTINGS
# ============================================================================

RECIPES_SPARK_LOG = os.path.join(LOGS_DIR, "recipes_extraction_spark.log")
RECIPES_MAPPING_FILE = os.path.join(INDEX_DIR, "mapping.jsonl")
RECIPES_SPARK_PARTITIONS = 500

# ============================================================================
# WIKI MAPPER SETTINGS
# ============================================================================

WIKI_DATA_DIR = os.path.join(DATA_DIR, "wiki_data")
WIKI_PAGES_DIR = os.path.join(WIKI_DATA_DIR, "pages")
WIKI_DUMP_FILE = "enwiki-20251020-pages-articles-multistream.xml.bz2"
WIKI_DUMP_PATH = os.path.join(WIKI_DATA_DIR, WIKI_DUMP_FILE)
WIKI_INDEX_FILE = "enwiki-20251020-pages-articles-multistream-index.txt.bz2"
WIKI_INDEX_PATH = os.path.join(WIKI_DATA_DIR, WIKI_INDEX_FILE)
WIKI_RECIPES_OUTPUT = os.path.join(INDEX_DIR, "wiki_recipes.jsonl")
WIKI_SPARK_LOG = os.path.join(LOGS_DIR, "wiki_mapper_spark.log")
WIKI_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "wiki_multistream")

# Matching parameters
WIKI_TITLE_WEIGHT = 0.3
WIKI_INGREDIENT_WEIGHT = 0.7
WIKI_MIN_MATCH_SCORE = 0.0  # Accept ALL matches (even low scores) to ensure every recipe gets matched

# ============================================================================
# REGEX PATTERNS
# ============================================================================

# URL patterns
RECIPE_URL_PATTERN = "/recipes/"
RECIPE_URL_MIN_SLASHES = 4

# HTML extraction patterns
URL_PATTERN = r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_chrome_options():
    """Get configured Chrome options for Selenium"""
    chrome_options = Options()

    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(f"--window-size={CHROME_WINDOW_SIZE}")
    chrome_options.add_argument(f"--user-agent={CHROME_USER_AGENT}")

    # Disable images and CSS for faster loading
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    return chrome_options


def setup_logging(log_file: str, log_level: int = LOG_LEVEL):
    """Setup logging configuration for a module"""
    import os

    os.makedirs(LOGS_DIR, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)
