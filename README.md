# Consulatation

1. inverse indexer moze byt tako ako ho mam (json) ? 
2. 2 metriky pri search ked budem vyberat napr. TF a DF (curry chicken) pre slovo alebo metrika ktora bude vzdy pritomna ? (vocab length/doc length) ci to je blbost parovat s TF
3. robots.txt ma 60 to je time.sleep(60) ?
4. spolu tam je nakoniec okolo 8.5k to moze byt ci este daco mam scrapnut ?

# Food Network UK Recipe Crawler and Search Engine

A Python-based web crawler and search engine for Food Network UK recipes. This project crawls recipe pages, scrapes structured data, builds a simple inverted index, and provides a command-line interface for full-text search.

## How It Works

The system is composed of four main modules that are run sequentially. The diagram below illustrates the workflow from crawling to searching.

### Architecture Diagram

```
[ Start URL: foodnetwork.co.uk ]
             |
             v
    +------------------+
    |   1. Crawler     |
    |  (src/crawler.py)|
    +------------------+
             |
             | Saves raw HTML files to data/raw_html/
             | and recipe URLs to data/urls.txt
             v
    +------------------+
    |   2. Scraper     |
    |  (src/scraper.py)|
    +------------------+
             |
             | Parses HTML to extract structured data,
             | saves it to data/scraped/recipes.jsonl
             v
    +------------------+
    |   3. Indexer     |
    |  (src/indexer.py)|
    +------------------+
             |
             | Builds an inverted index from recipe data,
             | saves it to data/index/search_index.pkl
             v
    +------------------+      +------------------+
    |   4. Searcher    |<---->|   User typing a  |
    |   (src/search.py)|      |      query       |
    +------------------+      +------------------+
             |
             v
    [Ranked Search Results]
```

### Component Descriptions

1.  **Crawler (`src/crawler.py`)**: Uses Selenium to dynamically browse `foodnetwork.co.uk`. It starts from a given URL, discovers all links on the page, and adds valid new ones to a queue. It proceeds in a Breadth-First Search (BFS) manner, saving the raw HTML of each visited recipe page to the `data/raw_html` directory. It also compiles a list of all found recipe URLs in `data/urls.txt`.

2.  **Scraper (`src/scraper.py`)**: Processes the HTML files saved by the crawler. It reads the list of recipe URLs and, for each one, parses the corresponding HTML file to extract structured information like title, description, ingredients, and method. This clean, structured data is then saved into a single JSONL file: `data/scraped/recipes.jsonl`.

3.  **Indexer (`src/indexer.py`)**: Reads the `recipes.jsonl` file and builds a simple inverted index. For each recipe, it tokenizes the text, removes common English stop words, and counts the frequency of each term (Term Frequency - TF). The final index, which maps terms to the documents they appear in, is saved as a compressed binary file at `data/index/search_index.pkl`.

4.  **Searcher (`src/search.py`)**: Loads the `search_index.pkl` file and provides an interactive command-line interface. When a user enters a query, the searcher tokenizes it, calculates TF-IDF scores to determine document relevance, and presents a ranked list of the top 10 matching recipes.

## Project Structure

```
├── data/
│   ├── index/         # Stores the final search_index.pkl
│   ├── raw_html/      # Stores raw HTML files from the crawler
│   └── scraped/       # Stores the scraped recipes.jsonl file
├── logs/              # Contains log files for each module
├── src/
│   ├── crawler.py     # Module 1: Crawls the website for recipe pages
│   ├── scraper.py     # Module 2: Extracts structured data from HTML
│   ├── indexer.py     # Module 3: Builds the inverted search index
│   └── search.py      # Module 4: Provides the search CLI
├── tests/
│   └── test.py        # Evaluates search engine performance
├── .gitignore
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- Google Chrome browser (for the Selenium crawler)

## Setup Instructions

### 1. Clone the Project

Ensure you have all the project files in your local workspace.

### 2. Create and Activate a Virtual Environment

Open a terminal or command prompt in the project's root directory.

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` at the beginning of your command prompt.

### 3. Install Dependencies

Install the required Python packages using pip:
```cmd
pip install -r requirements.txt
```
This will install `selenium` and other necessary libraries. The crawler also requires `chromedriver`, which `selenium` will attempt to download and manage automatically.

## How to Run the Project

Run the modules from the root directory in the following order.

### Step 1: Crawl the Website
This step uses Selenium to open a Chrome browser and download recipe pages. It can take a significant amount of time.
```cmd
python src/crawler.py
```
- **Output**: HTML files in `data/raw_html/` and a URL list in `data/urls.txt`.

### Step 2: Scrape Recipe Data
This step parses the downloaded HTML to extract structured recipe information.
```cmd
python src/scraper.py
```
- **Output**: A `recipes.jsonl` file in `data/scraped/`.

### Step 3: Build the Search Index
This step creates the inverted index from the scraped recipe data.
```cmd
python src/indexer.py
```
- **Output**: A `search_index.pkl` file in `data/index/`.

### Step 4: Search for Recipes
Once the index is built, you can start the interactive search engine.
```cmd
python src/search.py
```
- You will be prompted to enter search queries like "chicken soup" or "chocolate cake".
- Type `exit` or `quit` to close the program.

### (Optional) Step 5: Evaluate Search Performance
After building the index, you can run the evaluation test to measure the search engine's Hit Rate and Precision.
```cmd
python tests/test.py
```
- **Output**: A report with performance metrics based on 50 auto-generated queries.

## Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment:
```cmd
deactivate
```
