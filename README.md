# Consulatation

1. inverse indexer moze byt tako ako ho mam (json) ? 
2. 2 metriky pri search ked budem vyberat napr. TF a DF (curry chicken) pre slovo alebo metrika ktora bude vzdy pritomna ? (vocab length/doc length) ci to je blbost parovat s TF
3. robots.txt ma 60 to je time.sleep(60) ?
4. spolu tam je nakoniec okolo 8.5k to moze byt ci este daco mam scrapnut ?

# Food Network UK Recipe Crawler and Search Engine

A Python-based web crawler and search engine for Food Network UK recipes. This project crawls recipe pages, builds an inverted index, and provides fulltext search functionality with TF-IDF and BM25 scoring.

## Features

- **Web Crawler**: Crawls Food Network UK recipe collections and downloads HTML pages
- **Recipe Indexer**: Extracts recipe data and builds inverted index 
- **Search Engine**: Provides fulltext search functionality for recipes
- **Logging**: Comprehensive logging for debugging and monitoring

## Project Structure

```
├── crawler.py          # Web crawler for Food Network UK recipes
├── indexer.py          # Recipe indexer with TF-IDF and BM25 metrics
├── search.py           # Search engine with fulltext search
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Data directory
│   ├── raw_html/      # Downloaded HTML files
│   └── index/         # Generated index files
└── logs/              # Log files
    ├── crawler.log    # Crawler logs
    ├── indexer.log    # Indexer logs
    └── search.log     # Search logs
```

## Prerequisites

- Python 3.13

## Setup Instructions

### Step 1: Clone or Download the Project

If you haven't already, make sure you have all the project files in your workspace.

### Step 2: Create a Virtual Environment

Open Command Prompt (cmd) and navigate to your project directory:

```cmd
cd "c:\Users\risos\OneDrive\Desktop\FIIT\7. semester\VINF"
```

Create a virtual environment:

```cmd
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

Activate the virtual environment:

```cmd
.venv\Scripts\activate
```

You should see `(.venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### Step 4: Install Dependencies

Install the required Python packages:

```cmd
pip install -r requirements.txt
```

This will install:
- `requests` - For making HTTP requests to web pages
- `beautifulsoup4` - For parsing HTML content

## How to Run the Project

### Step 1: Crawl Recipe Data

Run the crawler to download recipe HTML pages from Food Network UK:

```cmd
python crawler.py
```

This will:
- Create necessary directories (`data/raw_html/`, `logs/`)
- Crawl Food Network UK recipe collections
- Download HTML pages to `data/raw_html/`
- Generate logs in `logs/crawler.log`

**Note**: The crawling process may take several minutes to hours depending on the number of recipes.

### Step 2: Build the Search Index

After crawling is complete, build the inverted index:

```cmd
python indexer.py
```

This will:
- Parse downloaded HTML files
- Extract recipe data (title, ingredients, method, chef name, time,...)
- Save index files to `data/index/`
- Generate logs in `logs/indexer.log`

### Step 3: Search Recipes

Once the index is built, you can search for recipes:

```cmd
python search.py
```

This will start an interactive search interface where you can:
- Enter search queries
- View search results with scores
- Browse recipe details

## Usage Examples

### Running the Full Pipeline

To run the complete pipeline from start to finish:

```cmd
# Activate virtual environment 
.venv\Scripts\activate

# Step 1: Crawl recipes
python crawler.py

# Step 2: Build index
python indexer.py

# Step 3: Search recipes
python search.py
```

### Sample Search Queries

Once the search engine is running, you can try queries like:
- "chicken pasta"
- "chocolate cake"
- "vegetarian"
- "Jamie Oliver" (chef name)
- "30 minutes" (preparation time)

## Output Files

After running the project, you'll find:

- `data/raw_html/*.html` - Downloaded recipe HTML files
- `data/index/recipe_documents.json` - Processed recipe documents
- `data/index/fulltext_index.json` - Fulltext search index
- `data/index/tfidf_index.json` - TF-IDF index (if generated)
- `data/index/bm25_index.json` - BM25 index (if generated)
- `logs/*.log` - Log files for debugging

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```cmd
deactivate
```

## Log Files

Check the log files in the `logs/` directory for detailed error information:
- `crawler.log` - Crawling issues
- `indexer.log` - Indexing issues  
- `search.log` - Search issues

## Project Dependencies

This project uses the following Python libraries:

- **requests**: HTTP library for making web requests
- **beautifulsoup4**: HTML parsing library
