# VINF - Food Network UK Recipe Search with Wikipedia Mapping

Information retrieval pipeline implementing web crawling, data extraction, PySpark distributed processing, and Wikipedia article mapping. Built for FIIT's VINF course assignment.

## Features

- **Headless crawler**: Breadth-first Selenium crawler with checkpointing and resume support
- **Structured extraction**: HTML to JSONL conversion capturing recipe metadata (ingredients, method, chef, timing, etc.)
- **PySpark indexing**: Distributed TF-IDF indexing with tokenization, stop word removal, and corpus statistics
- **Wikipedia mapping**: PySpark-based recipe-to-Wikipedia article mapping using ingredient similarity
- **Fault-tolerant processing**: Multi-stage checkpointing for safe interruption and resume
- **Dockerized deployment**: Complete environment with all dependencies pre-configured

## Quick Start (Docker - Recommended)

### 1. Build Docker Image
```bash
docker-compose build
```

### 2. Run Complete Pipeline
```bash
docker-compose run --rm recipe-processor python src/crawler.py

docker-compose run --rm recipe-processor python src/extractor.py

docker-compose run --rm recipe-processor python src/indexer.py
```

### 3. Download Wikipedia Dump
```bash
./scripts/download_wiki.sh
```

### 4. Map Recipes to Wikipedia
```bash
docker-compose run --rm wiki-processor python src/wiki_mapper_spark.py

docker-compose run --rm wiki-processor python src/wiki_mapper_spark.py
```

### 5. Generate Assignment Statistics
```bash
docker-compose run --rm wiki-processor python src/analyze_wiki_mapping.py
```

## Project Structure

```
VINF/
├── src/
│   ├── crawler.py              # Selenium-based web crawler
│   ├── extractor.py            # HTML → JSONL extraction
│   ├── indexer.py              # PySpark TF-IDF indexing
│   ├── wiki_mapper_spark.py    # Wikipedia mapping with PySpark
│   ├── analyze_wiki_mapping.py # Statistics generation
│   ├── spark_processing_demo.py # Demo for assignment presentation
│   ├── search.py               # Basic TF-IDF search
│   └── config.py               # Centralized configuration
├── scripts/
│   └── download_wiki.sh        # Wikipedia dump download
├── checkpoints/
│   └── wiki_mapping_checkpoint/ # PySpark checkpoints
├── data/
│   ├── raw_html/               # Crawled HTML files
│   ├── scraped/                # Extracted recipes (JSONL)
│   └── index/                  # Search index + Wikipedia mappings
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Service definitions
└── VINF_ASSIGNMENT.md          # Assignment documentation (Slovak)
```

## Pipeline Stages

### Stage 1: Crawl (src/crawler.py)
- Breadth-first search of Food Network UK
- Recipe URL filtering (`/recipes/{slug}` pattern)
- Checkpoint/resume support via pickle serialization
- Output: HTML files in `data/raw_html/`

### Stage 2: Extract (src/extractor.py or src/recipes_extraction_spark.py)
- HTML → Markdown conversion (MarkItDown library)
- Regex-based structured data extraction
- Output: `data/scraped/recipes.jsonl` (one recipe per line)

### Stage 3: Index (src/indexer.py)
- PySpark distributed processing
- Tokenization, stop word removal, TF-IDF calculation
- Output: `data/index/mapping.jsonl`, `index.jsonl`, `stats.jsonl`

### Stage 4: Wikipedia Mapping (src/wiki_mapper_spark.py)
- Processes 50GB+ Wikipedia XML dumps with PySpark
- Streams `.bz2` dumps directly in memory batches (size via `WIKI_STREAM_BATCH_SIZE`) without writing intermediate XML files; when `WIKI_DUMP_INDEX_PATH` points at the multistream index the mapper fans chunks across `WIKI_STREAM_WORKERS` parallel readers
- Threshold configurable via `WIKI_MIN_MATCH_SCORE` (defaults to 0.0 to keep the best candidate for every recipe)
- Debug option `WIKI_DEBUG_STOP_AFTER_FIRST_MATCH=true` halts right after the first recipe gets mapped (useful for inspecting early matches)
- Output written once per run to `WIKI_OUTPUT_PATH` (default `data/index/wiki_recipes.jsonl`) which mirrors the original recipe schema plus `wiki_title`, `wiki_url`, `wiki_ingredients`, and `match_score`
- Multi-source ingredient extraction:
  - 15+ infobox types (food, dish, beverage, cheese, bread, cake, dessert, soup, sauce, etc.)
  - Article sections (Ingredients, Recipe, Composition)
  - Category-based detection for pages without explicit ingredient lists
  - Handles nested templates: `{{plainlist|* item1\n* item2}}`
  - Handles inline format: `[[fish]], rice, salt`
- Ingredient-based Jaccard similarity matching
- Multi-stage checkpointing for fault tolerance
- Output: `data/index/wiki_recipes.jsonl`

### Stage 5: Analysis (src/analyze_wiki_mapping.py)
- Statistics on unique Wikipedia pages joined
- Top matched articles
- Extracted attributes demonstration
- Output: `data/index/wiki_analysis_summary.json`

## Docker Services

### recipe-processor
- Main pipeline service (crawl, extract, index)
- Memory limit: 4GB
- Mounts: data/, logs/, src/, scripts/

### wiki-processor
- Wikipedia processing with PySpark
- Memory limit: 8GB (configurable)
- Mounts: data/, logs/, src/, scripts/, checkpoints/ (persisted locally)

## Assignment Deliverables

This project implements the following VINF assignment requirements:

1. **Distributed Processing**: PySpark for indexing and Wikipedia mapping
2. **Wikipedia Integration**: 50GB+ XML dump processing with spark-xml
3. **REGEX Extraction**: 4 patterns for ingredient extraction from wikitext
4. **Join Operation**: Explode + inner join strategy for recipe-Wikipedia mapping
5. **Statistics**: Analysis of unique Wikipedia pages and extracted attributes

See `VINF_ASSIGNMENT.md` for detailed documentation (in Slovak).

## Configuration

All modules read from `src/config.py`:
- Paths: `DATA_DIR`, `RAW_HTML_DIR`, `SCRAPED_DIR`, `INDEX_DIR`
- Crawler: `START_URL`, `MAX_RETRIES`, `SKIP_EXTENSIONS`
- Indexer: `MIN_WORD_LENGTH`, `MAX_WORD_LENGTH`, `STOP_WORDS`
- Search: `DEFAULT_TOP_K` results per query

## Output Files

### Core Pipeline
- `data/raw_html/*.html` - Crawled HTML pages
- `data/scraped/recipes.jsonl` - Extracted recipes
- `data/index/mapping.jsonl` - Recipe metadata (doc_id → URL, title, etc.)
- `data/index/index.jsonl` - Inverted index (term → postings)
- `data/index/stats.jsonl` - Corpus statistics

### Wikipedia Mapping
- `data/index/wiki_recipes.jsonl` - Recipes with Wikipedia mappings
- `data/index/wiki_index.jsonl` - Copy of original index
- `data/index/wiki_stats.jsonl` - Copy of original stats
- `data/index/wiki_analysis_summary.json` - Assignment statistics

### Checkpoints (Persisted Locally)
- `data/crawler_checkpoint.pkl` - Crawler state (resume support)
- `checkpoints/wiki_multistream/` - Wikipedia mapping checkpoints (Parquet, JSONL, text)
  - `wiki_pages.parquet` - All Wikipedia pages with ingredients
  - `matches.jsonl` - Recipe-Wikipedia matches
  - `processed_streams.txt` - Stream offsets already processed
  - **Note:** Checkpoints are stored on your local machine and mounted into Docker containers