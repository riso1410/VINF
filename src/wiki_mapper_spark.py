import bz2
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Optional

from pyspark.sql import SparkSession

import config


# ============================================================================
# STEP 1: RECIPE NAME NORMALIZATION
# ============================================================================

def normalize_recipe_name(name: str) -> str:
    """
    Normalize recipe name for Wikipedia matching.

    Args:
        name: Recipe name to normalize

    Returns:
        Normalized recipe name
    """
    if not name:
        return ""

    normalized = name.lower().strip()
    normalized = re.sub(r'\([^)]*\)', '', normalized)  # Remove parentheticals
    normalized = re.sub(r"'s\b", '', normalized)  # Remove possessives
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)  # Keep alphanumeric
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Collapse spaces

    return normalized


def extract_recipe_type(title: str) -> str:
    """Extract recipe category from title."""
    title_lower = title.lower()

    categories = {
        'dessert': ['cake', 'cookie', 'pie', 'tart', 'pudding', 'ice cream', 'brownie'],
        'soup': ['soup', 'stew', 'chowder', 'bisque'],
        'salad': ['salad'],
        'pasta': ['pasta', 'spaghetti', 'linguine', 'fettuccine', 'lasagna'],
        'bread': ['bread', 'roll', 'bun', 'biscuit'],
        'beverage': ['drink', 'cocktail', 'smoothie', 'juice', 'tea', 'coffee'],
    }

    for category, keywords in categories.items():
        if any(keyword in title_lower for keyword in keywords):
            return category

    return 'main'


# ============================================================================
# STEP 2: WIKIPEDIA INDEX SEARCH
# ============================================================================

def parse_index_entry(line: str) -> tuple[int, int, str] | None:
    """Parse Wikipedia multistream index line."""
    if not line or not line.strip():
        return None

    try:
        parts = line.strip().split(':', 2)
        if len(parts) < 3:
            return None

        offset = int(parts[0])
        page_id = int(parts[1])
        title = parts[2]

        return (offset, page_id, title)
    except (ValueError, IndexError):
        return None


def calculate_similarity(recipe_name: str, wiki_title: str) -> float:
    """Calculate Jaccard similarity between recipe and Wikipedia title."""
    if not recipe_name or not wiki_title:
        return 0.0

    if recipe_name == wiki_title:
        return 1.0

    recipe_words = set(recipe_name.split())
    wiki_words = set(wiki_title.split())

    if not recipe_words or not wiki_words:
        return 0.0

    intersection = len(recipe_words & wiki_words)
    union = len(recipe_words | wiki_words)

    return intersection / union if union > 0 else 0.0


def find_wikipedia_matches(recipes: list[dict], logger) -> list[dict]:
    """
    Search Wikipedia index for recipe matches.

    Args:
        recipes: List of recipe dictionaries with 'url' and 'title'
        logger: Logger instance

    Returns:
        List of matched pages with offsets
    """
    logger.info("=" * 70)
    logger.info("STEP 1/3: SEARCHING WIKIPEDIA INDEX")
    logger.info("=" * 70)

    # Build lookup structures
    exact_lookup = {}
    fuzzy_lookup = defaultdict(list)

    for recipe in recipes:
        title = recipe.get('title', '')
        if not title:
            continue

        normalized = normalize_recipe_name(title)
        if not normalized:
            continue

        recipe_data = {
            'url': recipe.get('url', ''),
            'title': title,
            'normalized': normalized,
            'type': extract_recipe_type(title),
        }

        exact_lookup[normalized] = recipe_data

        # Index by first word for fuzzy matching
        first_word = normalized.split()[0] if normalized else ''
        if first_word and len(first_word) >= 3:
            fuzzy_lookup[first_word].append(recipe_data)

    logger.info(f"Prepared {len(recipes):,} recipes for matching")
    logger.info(f"  Exact lookup: {len(exact_lookup):,} entries")
    logger.info(f"  Fuzzy lookup: {len(fuzzy_lookup):,} first-word groups")

    # Search index
    index_path = config.WIKI_INDEX_PATH
    if not Path(index_path).exists():
        logger.error(f"Wikipedia index not found: {index_path}")
        raise FileNotFoundError(f"Wikipedia index not found: {index_path}")

    logger.info(f"\nSearching index: {index_path}")

    exact_matches = []
    fuzzy_matches = []
    processed_lines = 0

    with bz2.open(index_path, 'rt', encoding='utf-8') as f:
        for line in f:
            processed_lines += 1

            if processed_lines % 100000 == 0:
                total_found = len(exact_matches) + len(fuzzy_matches)
                logger.info(f"  Processed {processed_lines:,} entries, found {total_found:,} matches...")

            parsed = parse_index_entry(line)
            if not parsed:
                continue

            offset, page_id, wiki_title = parsed
            normalized_wiki = normalize_recipe_name(wiki_title)

            if not normalized_wiki:
                continue

            # Try exact match
            if normalized_wiki in exact_lookup:
                recipe_data = exact_lookup[normalized_wiki]
                exact_matches.append({
                    'offset': offset,
                    'page_id': page_id,
                    'wiki_title': wiki_title,
                    'recipe_url': recipe_data['url'],
                    'recipe_title': recipe_data['title'],
                    'recipe_type': recipe_data['type'],
                    'match_type': 'exact',
                    'similarity': 1.0
                })
                continue

            # Try fuzzy match
            first_word = normalized_wiki.split()[0] if normalized_wiki else ''
            if first_word in fuzzy_lookup:
                for candidate in fuzzy_lookup[first_word]:
                    similarity = calculate_similarity(candidate['normalized'], normalized_wiki)

                    if similarity >= 0.6:  
                        fuzzy_matches.append({
                            'offset': offset,
                            'page_id': page_id,
                            'wiki_title': wiki_title,
                            'recipe_url': candidate['url'],
                            'recipe_title': candidate['title'],
                            'recipe_type': candidate['type'],
                            'match_type': 'fuzzy',
                            'similarity': similarity
                        })
                        break

    all_matches = exact_matches + fuzzy_matches

    logger.info(f"\nProcessed {processed_lines:,} index entries")
    logger.info(f"  Exact matches: {len(exact_matches):,}")
    logger.info(f"  Fuzzy matches: {len(fuzzy_matches):,}")
    logger.info(f"  Total matches: {len(all_matches):,}")
    logger.info(f"  Match rate: {len(all_matches)/len(recipes)*100:.1f}%\n")

    return all_matches


# ============================================================================
# STEP 3: WIKIPEDIA PAGE EXTRACTION
# ============================================================================

def extract_ingredients_from_wikitext(wikitext: str) -> list[str]:
    """Extract ingredients from Wikipedia article wikitext."""
    if not wikitext:
        return []

    ingredients = set()

    # Food infobox pattern
    infobox_pattern = r'\{\{[Ii]nfobox\s+(?:food|dish|beverage|prepared food).*?\}\}'

    for infobox in re.findall(infobox_pattern, wikitext, re.DOTALL | re.IGNORECASE):
        ing_fields = re.findall(
            r'\|\s*(?:main_)?ingredient[s]?\s*=\s*([^\|]+)',
            infobox,
            re.IGNORECASE
        )

        for field in ing_fields:
            cleaned = re.sub(r'\[\[(?:[^\]]*\|)?([^\]]+)\]\]', r'\1', field)
            cleaned = re.sub(r'\{\{[^}]+\}\}', '', cleaned)
            cleaned = re.sub(r'<[^>]+>', '', cleaned)

            for part in re.split(r'[,\n]', cleaned):
                ingredient = part.strip().lower()
                if ingredient and 2 < len(ingredient) < 50:
                    ingredients.add(ingredient)

    return sorted(list(ingredients))[:30]


def extract_description_from_wikitext(wikitext: str) -> str:
    """Extract description from Wikipedia article lead section."""
    if not wikitext:
        return ""

    lead_match = re.search(r'^(.*?)(?:^==|\Z)', wikitext, re.MULTILINE | re.DOTALL)
    if not lead_match:
        return ""

    lead = lead_match.group(1)

    # Clean markup
    lead = re.sub(r'\{\{[^}]+\}\}', '', lead)
    lead = re.sub(r'<ref[^>]*>.*?</ref>', '', lead, flags=re.DOTALL)
    lead = re.sub(r'<!--.*?-->', '', lead, flags=re.DOTALL)
    lead = re.sub(r'\[\[(?:[^\]]*\|)?([^\]]+)\]\]', r'\1', lead)
    lead = re.sub(r"['\"]+", '', lead)
    lead = re.sub(r'<[^>]+>', '', lead)
    lead = re.sub(r'\s+', ' ', lead).strip()

    # Take first 500 chars
    if len(lead) > 500:
        lead = lead[:500].rsplit('.', 1)[0] + '.'

    return lead


def read_page_at_offset(dump_path: str, offset: int, target_title: str) -> Optional[str]:
    """Read Wikipedia page XML at specific byte offset."""
    try:
        with open(dump_path, 'rb') as f:
            f.seek(offset)

            decompressor = bz2.BZ2Decompressor()
            buffer = []
            max_read = 10 * 1024 * 1024  # 10MB limit
            bytes_read = 0

            while bytes_read < max_read:
                chunk = f.read(8192)
                if not chunk:
                    break

                try:
                    decompressed = decompressor.decompress(chunk)
                    buffer.append(decompressed.decode('utf-8', errors='ignore'))
                    bytes_read += len(chunk)

                    current_text = ''.join(buffer)
                    if '</page>' in current_text:
                        match = re.search(r'<page>.*?</page>', current_text, re.DOTALL)
                        if match:
                            page_xml = match.group(0)
                            if f'<title>{target_title}</title>' in page_xml:
                                return page_xml
                        break

                except EOFError:
                    break

        return None

    except Exception:
        return None


def process_matched_page(match_data: dict, dump_path: str) -> Optional[dict]:
    """Process a single matched Wikipedia page."""
    try:
        offset = match_data['offset']
        wiki_title = match_data['wiki_title']

        page_xml = read_page_at_offset(dump_path, offset, wiki_title)
        if not page_xml:
            return None

        try:
            root = ET.fromstring(page_xml)
        except ET.ParseError:
            return None

        text_elem = root.find('.//text')
        if text_elem is None or not text_elem.text:
            return None

        wikitext = text_elem.text

        return {
            'recipe_url': match_data['recipe_url'],
            'wiki_title': wiki_title,
            'wiki_url': f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}",
            'wiki_ingredients': extract_ingredients_from_wikitext(wikitext),
            'wiki_description': extract_description_from_wikitext(wikitext),
            'match_type': match_data['match_type'],
            'match_similarity': match_data['similarity'],
        }

    except Exception:
        return None


def process_offset_batch(batch_data: tuple) -> list[dict]:
    """Process all pages at a single offset."""
    offset, matches, dump_path = batch_data
    results = []

    for match in matches:
        result = process_matched_page(match, dump_path)
        if result:
            results.append(result)

    return results


def extract_wikipedia_pages(matches: list[dict], logger) -> list[dict]:
    """
    Extract Wikipedia pages using PySpark.

    Args:
        matches: List of matched pages with offsets
        logger: Logger instance

    Returns:
        List of extracted Wikipedia data
    """
    logger.info("=" * 70)
    logger.info("STEP 2/3: EXTRACTING WIKIPEDIA PAGES")
    logger.info("=" * 70)

    # Group by offset
    offset_groups = defaultdict(list)
    for match in matches:
        offset_groups[match['offset']].append(match)

    logger.info(f"Grouped {len(matches):,} matches into {len(offset_groups):,} unique offsets")

    # Check dump
    dump_path = config.WIKI_DUMP_PATH
    if not Path(dump_path).exists():
        logger.error(f"Wikipedia dump not found: {dump_path}")
        raise FileNotFoundError(f"Wikipedia dump not found: {dump_path}")

    # Initialize Spark
    logger.info("\nInitializing PySpark...")

    spark = SparkSession.builder \
        .appName("Wikipedia Recipe Extractor") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    try:
        # Prepare tasks
        tasks = [(offset, matches, dump_path) for offset, matches in offset_groups.items()]
        num_partitions = max(1, len(tasks) // 20)

        logger.info(f"Using {num_partitions} Spark partitions for {len(tasks):,} tasks")
        logger.info("Starting parallel extraction...\n")

        # Parallel extraction
        rdd = spark.sparkContext.parallelize(tasks, num_partitions)
        results_rdd = rdd.flatMap(process_offset_batch)
        extracted = results_rdd.collect()

        logger.info(f"Successfully extracted {len(extracted):,} Wikipedia pages")
        logger.info(f"Extraction success rate: {len(extracted)/len(matches)*100:.1f}%")

        # Statistics
        with_ingredients = sum(1 for r in extracted if r['wiki_ingredients'])
        logger.info(f"Pages with ingredients: {with_ingredients:,} ({with_ingredients/len(extracted)*100:.1f}%)\n")

        return extracted

    finally:
        spark.stop()


# ============================================================================
# STEP 4: MERGE RECIPES WITH WIKIPEDIA DATA
# ============================================================================

def merge_recipes_with_wikipedia(recipes: list[dict], wiki_data: list[dict], logger) -> list[dict]:
    """
    Merge recipes with Wikipedia data.

    Args:
        recipes: List of recipe dictionaries
        wiki_data: List of extracted Wikipedia data
        logger: Logger instance

    Returns:
        List of enriched recipes
    """
    logger.info("=" * 70)
    logger.info("STEP 3/3: MERGING RECIPES WITH WIKIPEDIA DATA")
    logger.info("=" * 70)

    # Build lookup by URL
    wiki_lookup = {}
    for wiki_item in wiki_data:
        url = wiki_item['recipe_url']
        wiki_lookup[url] = wiki_item

    # Merge
    enriched = []
    matched_count = 0

    for recipe in recipes:
        url = recipe.get('url', '')

        enriched_recipe = recipe.copy()

        if url in wiki_lookup:
            wiki_item = wiki_lookup[url]
            enriched_recipe.update({
                'wiki_title': wiki_item['wiki_title'],
                'wiki_url': wiki_item['wiki_url'],
                'wiki_ingredients': wiki_item['wiki_ingredients'],
                'wiki_description': wiki_item['wiki_description'],
            })
            matched_count += 1
        else:
            enriched_recipe.update({
                'wiki_title': '',
                'wiki_url': '',
                'wiki_ingredients': [],
                'wiki_description': '',
            })

        enriched.append(enriched_recipe)

    logger.info(f"Total recipes: {len(recipes):,}")
    logger.info(f"Matched with Wikipedia: {matched_count:,}")
    logger.info(f"Match rate: {matched_count/len(recipes)*100:.1f}%\n")

    return enriched


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution: complete Wikipedia enrichment pipeline."""
    logger = config.setup_logging(config.WIKI_SPARK_LOG)

    logger.info("=" * 70)
    logger.info("WIKIPEDIA RECIPE ENRICHMENT PIPELINE")
    logger.info("=" * 70)
    logger.info("")

    # Load recipes
    recipes_file = config.RECIPES_FILE
    if not Path(recipes_file).exists():
        logger.error(f"Recipes file not found: {recipes_file}")
        return

    logger.info(f"Loading recipes from: {recipes_file}")
    recipes = []

    with open(recipes_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                recipe = json.loads(line)
                if recipe.get('title'):
                    recipes.append(recipe)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(recipes):,} recipes\n")

    # Step 1: Find Wikipedia matches
    matches = find_wikipedia_matches(recipes, logger)

    if not matches:
        logger.warning("No Wikipedia matches found!")
        return

    # Step 2: Extract Wikipedia pages
    wiki_data = extract_wikipedia_pages(matches, logger)

    # Step 3: Merge
    enriched_recipes = merge_recipes_with_wikipedia(recipes, wiki_data, logger)

    # Save output
    output_file = config.WIKI_RECIPES_OUTPUT
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing output to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for recipe in enriched_recipes:
            json.dump(recipe, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saved {len(enriched_recipes):,} enriched recipes")

    # Show sample
    logger.info("Sample enriched recipes (first 3 with Wikipedia data):")
    sample_count = 0
    for recipe in enriched_recipes:
        if recipe.get('wiki_url'):
            logger.info(f"\n  Recipe: {recipe['title']}")
            logger.info(f"  Wikipedia: {recipe.get('wiki_title', 'N/A')}")
            ing_count = len(recipe.get('wiki_ingredients', []))
            logger.info(f"  Wikipedia ingredients: {ing_count}")
            sample_count += 1
            if sample_count >= 3:
                break

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
