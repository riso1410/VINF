import re
import glob
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import config

SIMILARITY_THRESHOLD = 0.45  # Levenshtein normalized similarity threshold

# ============================================================================
# RECIPE NAME NORMALIZATION
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
    normalized = re.sub(r'[^a-z\s]', ' ', normalized)  # Keep alpha
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Collapse spaces

    return normalized


# ============================================================================
# PYSPARK PREPROCESSING & TOKENIZATION
# ============================================================================

def preprocess_dataframe(df, title_col: str, id_col: str):
    """
    Clean, tokenize, and remove stopwords from a DataFrame.
    
    Args:
        df: Spark DataFrame with title column
        title_col: Name of the title column
        id_col: Name of the ID column
        
    Returns:
        Processed DataFrame with filtered tokens
    """
    # Lowercase and remove punctuation
    processed_df = df.withColumn(
        "cleaned_title",
        F.lower(F.regexp_replace(F.col(title_col), r"[^\w\s]", ""))
    )
    
    # Remove possessives and parentheticals
    processed_df = processed_df.withColumn(
        "cleaned_title",
        F.regexp_replace(F.col("cleaned_title"), r"\s*\([^)]*\)\s*", " ")
    )
    processed_df = processed_df.withColumn(
        "cleaned_title",
        F.regexp_replace(F.col("cleaned_title"), r"'s\b", "")
    )
    
    # Collapse multiple spaces
    processed_df = processed_df.withColumn(
        "cleaned_title",
        F.regexp_replace(F.col("cleaned_title"), r"\s+", " ")
    )
    processed_df = processed_df.withColumn(
        "cleaned_title",
        F.trim(F.col("cleaned_title"))
    )
    
    # Split into words (tokens)
    processed_df = processed_df.withColumn(
        "tokens",
        F.split(F.col("cleaned_title"), r"\s+")
    )
    
    # Only remove VERY common words that add no value
    # Keep words like "chicken", "soup", etc. - they're important!
    minimal_stopwords = list(config.STOP_WORDS)  # Convert set to list
    
    # Remove stopwords using array_except (native Spark function - no SQL parsing!)
    # First remove empty strings, then remove stopwords
    processed_df = processed_df.withColumn(
        "filtered_tokens",
        F.array_except(
            F.filter(F.col("tokens"), lambda x: x != ""),  # Remove empty strings
            F.array(*[F.lit(word) for word in minimal_stopwords])  # Remove stopwords
        )
    )
    
    # Keep only ID, original title, cleaned title, and filtered tokens
    return processed_df.select(
        F.col(id_col),
        F.col(title_col),
        "cleaned_title",
        "filtered_tokens"
    )

# ============================================================================
# WIKIPEDIA INDEX LOADING
# ============================================================================

def load_wikipedia_index(spark, index_path: Path, logger):
    """
    Load Wikipedia multistream index into Spark DataFrame using native Spark operations.
    
    Args:
        spark: SparkSession
        index_path: Path to Wikipedia index file (.bz2)
        logger: Logger instance
        
    Returns:
        Spark DataFrame with columns: offset, page_id, wiki_title
    """
    logger.info(f"Loading Wikipedia index: {index_path}")
    
    # Use Spark to read the bz2 file directly (Spark handles decompression automatically)
    text_rdd = spark.sparkContext.textFile(str(index_path))
    
    # Parse each line in parallel
    def parse_line(line):
        """Parse Wikipedia index line format: offset:page_id:title"""
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
    
    # Parse and filter out invalid lines
    parsed_rdd = text_rdd.map(parse_line).filter(lambda x: x is not None)
    
    # Create DataFrame
    wiki_df = spark.createDataFrame(
        parsed_rdd,
        ["offset", "page_id", "wiki_title"]
    )
    
    # Cache the DataFrame for better performance in subsequent operations
    wiki_df = wiki_df.cache()
    
    # Trigger computation and count to show progress
    count = wiki_df.count()
    logger.info(f"Loaded {count:,} Wikipedia index entries")
    
    return wiki_df


# ============================================================================
# PYSPARK FUZZY MATCHING PIPELINE
# ============================================================================

def find_wikipedia_matches_spark(spark, recipes_df, wiki_df, logger):
    """
    Find fuzzy matches between recipes and Wikipedia titles using token-based blocking.
    
    Strategy: Token-based blocking with Levenshtein similarity
    
    Args:
        spark: SparkSession
        recipes_df: DataFrame with ALL recipe columns (url, title, description, ingredients, etc.)
        wiki_df: DataFrame with columns: offset, page_id, wiki_title
        logger: Logger instance
        
    Returns:
        DataFrame with ALL recipe columns + Wikipedia data
    """
    logger.info("=" * 70)
    logger.info("TOKEN-BASED FUZZY MATCHING PIPELINE")
    logger.info("=" * 70)
    
    # --- Step 1: Preprocess both DataFrames ---
    logger.info("\nStep 1: Preprocessing recipes...")
    
    # Deduplicate recipes by URL first (keep first occurrence)
    recipes_df = recipes_df.dropDuplicates(["url"])
    recipe_count = recipes_df.count()
    logger.info(f"Unique recipes after deduplication: {recipe_count:,}")
    
    recipes_processed = preprocess_dataframe(
        recipes_df, 
        title_col="title", 
        id_col="url"
    )
    
    # Join back ALL original recipe columns
    # Get column names from processed DataFrame
    proc_cols = recipes_processed.columns  # [url, title, cleaned_title, filtered_tokens]
    other_cols = [c for c in recipes_df.columns if c not in ["url", "title"]]
    
    recipes_processed = recipes_processed.alias("proc").join(
        recipes_df.select("url", *other_cols).alias("orig"),
        F.col("proc.url") == F.col("orig.url")
    ).select(
        *[F.col(f"proc.{c}") for c in proc_cols],  # All columns from processed
        *[F.col(f"orig.{c}") for c in other_cols]  # Other columns from original
    )
    
    recipes_processed = recipes_processed.withColumnRenamed("url", "recipe_url")
    recipes_processed = recipes_processed.withColumnRenamed("title", "recipe_title")
        
    logger.info("\nStep 2: Preprocessing Wikipedia titles...")
    wiki_processed = preprocess_dataframe(
        wiki_df,
        title_col="wiki_title",
        id_col="page_id"
    )
    # Keep offset for later use
    wiki_processed = wiki_processed.join(
        wiki_df.select("page_id", "offset"),
        "page_id"
    )
    
    logger.info(f"Processed {wiki_processed.count():,} Wikipedia entries")
    
    # --- Token-based Blocking with Fuzzy Matching ---
    logger.info("\n" + "=" * 70)
    logger.info("TOKEN-BASED BLOCKING + FUZZY MATCHING")
    logger.info("=" * 70)
    
    # Get all recipe columns except the processing ones
    recipe_data_cols = [c for c in recipes_processed.columns 
                       if c not in ['cleaned_title', 'filtered_tokens']]
    
    # Explode recipes on filtered tokens
    recipes_exploded = recipes_processed.select(
        "recipe_url",
        "recipe_title",
        "cleaned_title",
        "filtered_tokens",
        F.explode("filtered_tokens").alias("token")
    ).filter(F.col("token") != "")
    
    logger.info(f"Recipe tokens: {recipes_exploded.count():,}")
    
    # Explode Wikipedia on filtered tokens
    wiki_exploded = wiki_processed.select(
        "page_id",
        "wiki_title",
        "cleaned_title",
        "filtered_tokens",
        "offset",
        F.explode("filtered_tokens").alias("token")
    ).filter(F.col("token") != "")
    
    logger.info(f"Wikipedia tokens: {wiki_exploded.count():,}")
    
    # Join on token to find candidates (blocking step)
    logger.info("\nJoining on shared tokens...")
    
    # Alias the DataFrames to avoid column ambiguity
    recipes_aliased = recipes_exploded.alias("r")
    wiki_aliased = wiki_exploded.alias("w")
    
    candidate_pairs = recipes_aliased.join(
        wiki_aliased, 
        "token"
    ).select(
        F.col("r.recipe_url"),
        F.col("r.recipe_title"),
        F.col("r.cleaned_title").alias("recipe_cleaned"),
        F.col("r.filtered_tokens").alias("recipe_tokens"),
        F.col("w.page_id"),
        F.col("w.wiki_title"),
        F.col("w.cleaned_title").alias("wiki_cleaned"),
        F.col("w.filtered_tokens").alias("wiki_tokens"),
        F.col("w.offset")
    )
    
    candidate_count = candidate_pairs.count()
    logger.info(f"Candidate pairs: {candidate_count:,}")
    
    # Calculate Levenshtein similarity on cleaned titles
    logger.info("\nCalculating Levenshtein similarity on cleaned titles...")
    
    matches_with_distance = candidate_pairs.withColumn(
        "distance",
        F.levenshtein(F.col("recipe_cleaned"), F.col("wiki_cleaned"))
    )
    
    # Calculate normalized similarity score (0.0 to 1.0)
    fuzzy_prelim = matches_with_distance.withColumn(
        "similarity",
        1 - (
            F.col("distance") / 
            F.greatest(
                F.length("recipe_cleaned"), 
                F.length("wiki_cleaned")
            )
        )
    ).filter(
        F.col("similarity") >= SIMILARITY_THRESHOLD
    )
    
    # Join back ALL recipe columns using aliases
    fuzzy_aliased = fuzzy_prelim.alias("fuzz")
    recipes_aliased = recipes_processed.alias("rec")
    
    all_matches = fuzzy_aliased.join(
        recipes_aliased,
        F.col("fuzz.recipe_url") == F.col("rec.recipe_url")
    ).select(
        # All recipe columns from recipes_processed
        *[F.col(f"rec.{c}") for c in recipe_data_cols],
        # Wikipedia data from fuzzy_prelim
        F.col("fuzz.wiki_title"),
        # Keep similarity for ranking
        F.col("fuzz.similarity"),
        # Create Wikipedia URL from title
        F.concat(
            F.lit("https://en.wikipedia.org/wiki/"),
            F.regexp_replace(F.col("fuzz.wiki_title"), r" ", "_")
        ).alias("wiki_url"),
        # Keep offset for description extraction
        F.col("fuzz.offset").alias("wiki_offset")
    )
    
    fuzzy_count = all_matches.count()
    logger.info(f"Matches above threshold ({SIMILARITY_THRESHOLD}): {fuzzy_count:,}")
    
    # --- Final: Keep best match per recipe ---
    logger.info("\n" + "=" * 70)
    logger.info("FINAL: Selecting best match per recipe")
    logger.info("=" * 70)
    
    from pyspark.sql.window import Window
    
    window_spec = Window.partitionBy("recipe_url").orderBy(F.col("similarity").desc())
    
    best_matches = all_matches.withColumn(
        "rank",
        F.row_number().over(window_spec)
    ).filter(
        F.col("rank") == 1
    ).drop("rank", "similarity")  # Drop rank and similarity    
    # Left join: Keep ALL recipes from recipes_processed
    all_recipes_with_wiki = recipes_processed.alias("all_rec").join(
        best_matches.alias("matched"),
        F.col("all_rec.recipe_url") == F.col("matched.recipe_url"),
        "left"
    ).select(
        # All original recipe columns from recipes_processed
        *[F.col(f"all_rec.{c}") for c in recipe_data_cols],
        # Wikipedia columns from best_matches (will be null for non-matches)
        F.col("matched.wiki_title"),
        F.col("matched.wiki_url"),
        F.col("matched.wiki_offset")
    )
    
    # Add wiki_description placeholder (will be filled by enrich_with_descriptions)
    all_recipes_with_wiki = all_recipes_with_wiki.withColumn(
        "wiki_description",
        F.lit(None).cast("string")
    )
    
    # Calculate match rate
    logger.info("")
    
    return all_recipes_with_wiki


# ============================================================================
# WIKIPEDIA DESCRIPTION EXTRACTION
# ============================================================================

def enrich_with_descriptions(spark, matches_df, dump_path: Path, logger):
    """
    Enrich matches DataFrame with Wikipedia descriptions using optimized batch processing.
    
    Reads the dump file once in sorted order for maximum efficiency.
    
    Args:
        spark: SparkSession
        matches_df: DataFrame with matches (must have wiki_offset column)
        dump_path: Path to Wikipedia dump file
        logger: Logger instance
        
    Returns:
        DataFrame with wiki_description filled in
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING WIKIPEDIA DESCRIPTIONS")
    logger.info("=" * 70)
    logger.info(f"Dump file: {dump_path}")
    
    if not dump_path.exists():
        logger.warning(f"Wikipedia dump not found at {dump_path}")
        logger.warning("Skipping description extraction")
        return matches_df
    
    # Filter out rows with null offsets
    logger.info("Filtering recipes with Wikipedia matches...")
    matches_with_offset = matches_df.filter(F.col("wiki_offset").isNotNull())
    
    matched_count = matches_with_offset.count()
    total_count = matches_df.count()
    logger.info(f"Found {matched_count:,} recipes with Wikipedia matches (out of {total_count:,} total)")
        
    matches_with_offset = matches_with_offset.orderBy("wiki_offset")
    
    # Collect offsets and other data to driver (trade memory for speed)
    logger.info("Collecting match data to driver for batch processing...")
    matches_collected = matches_with_offset.select("recipe_url", "wiki_offset").collect()
    offset_to_url = {row.wiki_offset: row.recipe_url for row in matches_collected}
    
    logger.info(f"Processing {len(offset_to_url)} unique offsets...")
    
    import re
    import bz2
    
    # Compile regex patterns once
    first_section_pattern = re.compile(
        r'={2,}\s*([^=\n]+?)\s*={2,}\s*(.*?)(?:={2,}|</text>)',
        re.DOTALL
    )
    refs_pattern = re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL)
    templates_pattern = re.compile(r'{{[^}]*}}')
    wiki_links_pattern = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')
    bold_italic_pattern = re.compile(r"'''?([^']+)'''?")
    html_tags_pattern = re.compile(r'<[^>]+>')
    multiple_spaces_pattern = re.compile(r'\s+')
    
    def extract_clean_description(article_xml):
        """Extract and clean description from article XML"""
        match = first_section_pattern.search(article_xml)
        if not match:
            return None
        
        description = match.group(2)
        description = refs_pattern.sub('', description)
        description = templates_pattern.sub('', description)
        description = wiki_links_pattern.sub(r'\1', description)
        description = bold_italic_pattern.sub(r'\1', description)
        description = html_tags_pattern.sub('', description)
        description = multiple_spaces_pattern.sub(' ', description)
        description = description.strip()
        
        return description if description else None
    
    # Read dump file once, extract all needed descriptions
    logger.info("Reading Wikipedia dump using multistream seek optimization...")
    descriptions = {}
    sorted_offsets = sorted(offset_to_url.keys())
    processed = 0
    
    # Process each offset by seeking directly to it
    with open(dump_path, 'rb') as raw_file:
        for offset in sorted_offsets:
            try:
                # Seek to the multistream block offset
                raw_file.seek(offset)
                
                # Create a bz2 decompressor for this block
                decompressor = bz2.BZ2Decompressor()
                article_xml = None
                
                # Read and decompress until we get a complete article
                article_lines = []
                while True:
                    chunk = raw_file.read(8192)  # Read in chunks
                    if not chunk:
                        break
                    
                    try:
                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            text = decompressed.decode('utf-8', errors='ignore')
                            article_lines.append(text)
                            
                            # Check if we have a complete article
                            if '</page>' in text:
                                article_xml = ''.join(article_lines)
                                break
                    except EOFError:
                        # End of this compressed block
                        break
                
                if article_xml:
                    description = extract_clean_description(article_xml)
                    descriptions[offset] = description
                    
                    processed += 1
                    if processed % 1000 == 0:
                        logger.info(f"  Extracted {processed}/{len(sorted_offsets)} descriptions...")
                
            except Exception as e:
                logger.warning(f"  Failed to extract at offset {offset}: {e}")
                continue
    
    logger.info(f"Extracted {len(descriptions)} descriptions from dump")
    
    # Create lookup DataFrame
    description_data = [
        (offset, desc) 
        for offset, desc in descriptions.items()
    ]
    
    descriptions_df = spark.createDataFrame(
        description_data,
        ["wiki_offset", "wiki_description"]
    )
    
    # Join descriptions back to matches
    logger.info("Joining descriptions back to matches...")
    # Drop the placeholder wiki_description column first to avoid ambiguity
    enriched_df = matches_df.drop("wiki_description").join(
        descriptions_df,
        "wiki_offset",
        "left"
    ).drop("wiki_offset")
    
    # Count successful extractions
    desc_count = enriched_df.filter(F.col("wiki_description").isNotNull()).count()
    total_count = enriched_df.count()
    
    logger.info(f"Extracted {desc_count:,} descriptions out of {total_count:,} matches")
    logger.info(f"Success rate: {desc_count/total_count*100:.1f}%")
    logger.info("")
    
    return enriched_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution: Wikipedia fuzzy title matching using PySpark."""
    logger = config.setup_logging(config.WIKI_SPARK_LOG)

    logger.info("=" * 70)
    logger.info("WIKIPEDIA RECIPE MATCHING")
    logger.info("=" * 70)
    logger.info("")

    # --- Initialize Spark ---
    logger.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Wikipedia Recipe Matcher") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    try:
        # --- Load recipes ---
        recipes_file = config.RECIPES_FILE
        if not Path(recipes_file).exists():
            logger.error(f"Recipes file not found: {recipes_file}")
            return

        logger.info(f"Loading recipes from: {recipes_file}")
        
        # Read JSONL file with ALL columns
        recipes_df = spark.read.json(str(recipes_file))
        
        # Filter only recipes with title and url
        recipes_df = recipes_df.filter(
            (F.col("title").isNotNull()) & 
            (F.col("url").isNotNull())
        )
        
        recipe_count = recipes_df.count()
        logger.info(f"Loaded {recipe_count:,} recipes")

        # --- Load Wikipedia index ---
        index_path = Path(config.WIKI_INDEX_PATH)
        if not index_path.exists():
            logger.error(f"Wikipedia index not found: {index_path}")
            return

        wiki_df = load_wikipedia_index(spark, index_path, logger)
        logger.info("")

        # --- Find matches ---
        matches_df = find_wikipedia_matches_spark(spark, recipes_df, wiki_df, logger)

        # --- Save matches WITHOUT descriptions first ---
        matches_output = Path(config.INDEX_DIR) / "wiki_matches.jsonl"
        logger.info(f"Saving matches without descriptions to: {matches_output}")
        
        # Drop wiki_description placeholder before saving
        matches_no_desc = matches_df.drop("wiki_description")
        
        temp_output = str(matches_output) + ".tmp"
        matches_no_desc.write.mode("overwrite").json(temp_output)
        
        logger.info("Combining output files...")
        with open(matches_output, 'w', encoding='utf-8') as outfile:
            part_files = sorted(glob.glob(f"{temp_output}/part-*.json"))
            for part_file in part_files:
                with open(part_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
        shutil.rmtree(temp_output)
        
        match_count = matches_df.count()
        logger.info(f"Saved {match_count:,} matches without descriptions")
        logger.info("")

        # --- Enrich with descriptions ---
        dump_path = Path(config.WIKI_DUMP_PATH)
        enriched_df = enrich_with_descriptions(spark, matches_df, dump_path, logger)

        # --- Save enriched matches WITH descriptions (overwrite) ---
        logger.info(f"Saving enriched matches WITH descriptions to: {matches_output}")

        temp_output = str(matches_output) + ".tmp"
        enriched_df.write.mode("overwrite").json(temp_output)
        
        logger.info("Combining output files...")
        with open(matches_output, 'w', encoding='utf-8') as outfile:
            part_files = sorted(glob.glob(f"{temp_output}/part-*.json"))
            for part_file in part_files:
                with open(part_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
        shutil.rmtree(temp_output)
        
        logger.info(f"Saved {match_count:,} enriched matches")
        logger.info("")
        logger.info("")

        # --- Show sample matches (top 10 by similarity) ---
        logger.info("\n" + "=" * 70)
        logger.info("MATCHING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nOutput file: {matches_output}")
        logger.info(f"This file maps {match_count:,} recipes to Wikipedia pages")
        logger.info("")

    finally:
        # Stop Spark session
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Done.")


if __name__ == "__main__":
    main()
