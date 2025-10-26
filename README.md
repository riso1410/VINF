# VINF

Pipeline for harvesting, normalizing, indexing, and searching Food Network UK recipes. The project combines a Selenium crawler, HTML-to-JSONL extractor, PySpark indexer, and CLI search client using 2 IDF methods.

## Features
- **Headless crawl**: Breadth-first Selenium crawler with checkpointing, recipe URL filtering, and optional sitemap reconciliation (`--xml`).
- **Structured scrape**: HTML extraction to JSON Lines, capturing titles, ingredients, method, chef, difficulty, timing, and servings.
- **TF-IDF indexing**: PySpark job that tokenizes (word-based) content, removes stop words, stores inverted index, document stats, and corpus metrics.
- **Interactive search**: Terminal UI supporting Robertson (BM25-style) and classical IDF scoring.

## Prerequisites
- Python 3.11.* (match the version configured for PySpark).  
- Google Chrome and a compatible ChromeDriver discoverable on `PATH`.  
- Java 8+ runtime for Spark; set `JAVA_HOME`/`SPARK_HOME` if not globally installed.  

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Ensure `chromedriver` matches your Chrome version; adjust the `PATH` or pass a custom location via Selenium environment variables if needed.

## Workflow
1. **Crawl** – collect HTML:  
   `python src/crawler.py` (add `--xml` to compare against `sitemap.xml` and download missing recipes).  
2. **Scrape** – convert HTML to structured JSONL:  
   `python src/extractor.py`
3. **Index** – build the TF-IDF search index with PySpark:  
   `python src/indexer.py`
4. **Search** – query the index interactively:  
   `python src/search.py` → type `change` to toggle between Robertson and classic IDF.  
Intermediate artifacts live in `data/` and are safe to regenerate; logs are emitted to `logs/`.

## Project Layout
- `src/` – core modules (`crawler.py`, `extractor.py`, `indexer.py`, `search.py`, `config.py`).  
- `data/` – pipeline outputs (`raw_html/`, `scraped/`, `index/`, checkpoints).  
- `logs/` – per-module log files configured in `config.py`.  
- `crawler.mmd` / `crawler.svg` – architecture diagrams for documentation.

## Configuration
Edit `src/config.py` to change the start URL, directory paths, Selenium chrome options, retry limits, stop-word list, and search defaults. All modules read from this single source, so adjust there rather than hard-coding values elsewhere.
