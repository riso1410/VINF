import glob
import re
import shutil
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import config


def normalize_recipe_name(name: str) -> str:
    if not name:
        return ""

    normalized = name.lower().strip()
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"'s\b", "", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# Clean and tokenize titles using Spark functions
def preprocess_dataframe(df, title_col: str, id_col: str):
    processed_df = df.withColumn("cleaned_title", F.lower(F.col(title_col)))

    processed_df = processed_df.withColumn(
        "cleaned_title",
        F.regexp_replace(F.col("cleaned_title"), r"\s*\([^)]*\)\s*", " "),
    )

    processed_df = processed_df.withColumn(
        "cleaned_title", F.regexp_replace(F.col("cleaned_title"), r"'s\b", "")
    )

    processed_df = processed_df.withColumn(
        "cleaned_title", F.regexp_replace(F.col("cleaned_title"), r"[^\w\s]", "")
    )

    processed_df = processed_df.withColumn(
        "cleaned_title", F.trim(F.regexp_replace(F.col("cleaned_title"), r"\s+", " "))
    )

    processed_df = processed_df.withColumn(
        "tokens", F.split(F.col("cleaned_title"), r"\s+")
    )

    minimal_stopwords = list(config.STOP_WORDS)

    # Remove stopwords using native Spark array functions
    processed_df = processed_df.withColumn(
        "filtered_tokens",
        F.array_except(
            F.filter(F.col("tokens"), lambda x: x != ""),
            F.array(*[F.lit(word) for word in minimal_stopwords]),
        ),
    )

    return processed_df.select(
        F.col(id_col), F.col(title_col), "cleaned_title", "filtered_tokens"
    )


# Load Wikipedia index into Spark DataFrame
def load_wikipedia_index(spark, index_path: Path, logger):
    logger.info(f"Loading Wikipedia index: {index_path}")

    text_rdd = spark.sparkContext.textFile(str(index_path))

    def parse_line(line):
        if not line or not line.strip():
            return None
        try:
            parts = line.strip().split(":", 2)
            if len(parts) < 3:
                return None
            offset = int(parts[0])
            page_id = int(parts[1])
            title = parts[2]
            return (offset, page_id, title)
        except (ValueError, IndexError):
            return None

    parsed_rdd = text_rdd.map(parse_line).filter(lambda x: x is not None)

    wiki_df = spark.createDataFrame(parsed_rdd, ["offset", "page_id", "wiki_title"])

    wiki_df = wiki_df.cache()

    count = wiki_df.count()
    logger.info(f"Loaded {count:,} Wikipedia index entries")

    return wiki_df


# Fuzzy match recipes with Wikipedia titles using Spark joins and token blocking
def find_wikipedia_matches_spark(spark, recipes_df, wiki_df, logger):
    logger.info("=" * 70)
    logger.info("TOKEN-BASED FUZZY MATCHING PIPELINE")
    logger.info("=" * 70)

    logger.info("\nStep 1: Preprocessing recipes...")

    recipes_df = recipes_df.dropDuplicates(["url"])
    recipe_count = recipes_df.count()
    logger.info(f"Unique recipes after deduplication: {recipe_count:,}")

    recipes_processed = preprocess_dataframe(
        recipes_df, title_col="title", id_col="url"
    )

    # Join back ALL original recipe columns
    proc_cols = recipes_processed.columns
    other_cols = [c for c in recipes_df.columns if c not in ["url", "title"]]

    recipes_processed = (
        recipes_processed.alias("proc")
        .join(
            recipes_df.select("url", *other_cols).alias("orig"),
            F.col("proc.url") == F.col("orig.url"),
        )
        .select(
            *[F.col(f"proc.{c}") for c in proc_cols],
            *[F.col(f"orig.{c}") for c in other_cols],
        )
    )

    recipes_processed = recipes_processed.withColumnRenamed("url", "recipe_url")
    recipes_processed = recipes_processed.withColumnRenamed("title", "recipe_title")

    logger.info("\nStep 2: Preprocessing Wikipedia titles...")
    wiki_processed = preprocess_dataframe(
        wiki_df, title_col="wiki_title", id_col="page_id"
    )

    logger.info("Filtering for food-related pages...")
    food_keywords_list = list(config.FOOD_KEYWORDS)

    wiki_processed = wiki_processed.filter(
        F.size(
            F.array_intersect(
                F.col("filtered_tokens"),
                F.array(*[F.lit(w) for w in food_keywords_list]),
            )
        )
        > 0
    )

    logger.info(f"Filtered to {wiki_processed.count():,} food-related entries")
    wiki_processed = wiki_processed.join(wiki_df.select("page_id", "offset"), "page_id")

    logger.info(f"Processed {wiki_processed.count():,} Wikipedia entries")

    logger.info("\n" + "=" * 70)
    logger.info("TOKEN-BASED BLOCKING + FUZZY MATCHING")
    logger.info("=" * 70)

    recipe_data_cols = [
        c
        for c in recipes_processed.columns
        if c not in ["cleaned_title", "filtered_tokens"]
    ]

    # Explode recipes on filtered tokens
    recipes_exploded = recipes_processed.select(
        "recipe_url",
        "recipe_title",
        "cleaned_title",
        "filtered_tokens",
        F.explode("filtered_tokens").alias("token"),
    ).filter(F.col("token") != "")

    logger.info(f"Recipe tokens: {recipes_exploded.count():,}")

    # Explode Wikipedia on filtered tokens
    wiki_exploded = wiki_processed.select(
        "page_id",
        "wiki_title",
        "cleaned_title",
        "filtered_tokens",
        "offset",
        F.explode("filtered_tokens").alias("token"),
    ).filter(F.col("token") != "")

    logger.info(f"Wikipedia tokens: {wiki_exploded.count():,}")

    logger.info("\nJoining on shared tokens...")

    recipes_aliased = recipes_exploded.alias("r")
    wiki_aliased = wiki_exploded.alias("w")

    # Join on token to find candidates (blocking step)
    candidate_pairs = recipes_aliased.join(wiki_aliased, "token").select(
        F.col("r.recipe_url"),
        F.col("r.recipe_title"),
        F.col("r.cleaned_title").alias("recipe_cleaned"),
        F.col("r.filtered_tokens").alias("recipe_tokens"),
        F.col("w.page_id"),
        F.col("w.wiki_title"),
        F.col("w.cleaned_title").alias("wiki_cleaned"),
        F.col("w.filtered_tokens").alias("wiki_tokens"),
        F.col("w.offset"),
    )

    candidate_count = candidate_pairs.count()
    logger.info(f"Candidate pairs: {candidate_count:,}")

    logger.info("\nCalculating Levenshtein similarity on cleaned titles...")

    matches_with_distance = candidate_pairs.withColumn(
        "distance", F.levenshtein(F.col("recipe_cleaned"), F.col("wiki_cleaned"))
    )

    fuzzy_prelim = matches_with_distance.withColumn(
        "similarity",
        1
        - (
            F.col("distance")
            / F.greatest(F.length("recipe_cleaned"), F.length("wiki_cleaned"))
        ),
    )

    fuzzy_aliased = fuzzy_prelim.alias("fuzz")
    recipes_aliased = recipes_processed.alias("rec")

    all_matches = fuzzy_aliased.join(
        recipes_aliased, F.col("fuzz.recipe_url") == F.col("rec.recipe_url")
    ).select(
        *[F.col(f"rec.{c}") for c in recipe_data_cols],
        F.col("fuzz.wiki_title"),
        F.col("fuzz.similarity"),
        F.concat(
            F.lit("https://en.wikipedia.org/wiki/"),
            F.regexp_replace(
                F.regexp_replace(F.col("fuzz.wiki_title"), r"[^a-zA-Z0-9\s\-\(\)]", ""),
                r" ",
                "_",
            ),
        ).alias("wiki_url"),
        F.col("fuzz.offset").alias("wiki_offset"),
        F.col("fuzz.page_id"),
    )

    fuzzy_count = all_matches.count()
    logger.info(f"Matches Count: {fuzzy_count:,}")

    logger.info("\n" + "=" * 70)
    logger.info("FINAL: Selecting TOP 3 matches per recipe")
    logger.info("=" * 70)

    from pyspark.sql.window import Window

    window_spec = Window.partitionBy("recipe_url").orderBy(F.col("similarity").desc())

    top_matches = all_matches.withColumn(
        "rank", F.row_number().over(window_spec)
    ).filter(F.col("rank") <= 3)

    # Left join: Keep ALL recipes from recipes_processed
    all_recipes_with_wiki = (
        recipes_processed.alias("all_rec")
        .join(
            top_matches.alias("matched"),
            F.col("all_rec.recipe_url") == F.col("matched.recipe_url"),
            "left",
        )
        .select(
            *[F.col(f"all_rec.{c}") for c in recipe_data_cols],
            F.col("matched.wiki_title"),
            F.col("matched.wiki_url"),
            F.col("matched.wiki_offset"),
            F.col("matched.page_id"),
            F.col("matched.rank"),
            F.col("matched.similarity"),
        )
    )

    all_recipes_with_wiki = all_recipes_with_wiki.withColumn(
        "wiki_description", F.lit(None).cast("string")
    )

    return all_recipes_with_wiki


def extract_wiki_data_from_xml(article_xml):
    import re

    lead_section_pattern = re.compile(r"^(.*?)(?=\n={2,})", re.DOTALL)
    redirect_pattern = re.compile(r"#REDIRECT\s*\[\[([^\]]+)\]\]", re.IGNORECASE)

    refs_pattern = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
    templates_pattern = re.compile(r"{{[^}]*}}", re.DOTALL)
    wiki_links_pattern = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
    bold_italic_pattern = re.compile(r"'''?([^']+)'''?")
    html_tags_pattern = re.compile(r"<[^>]+>")
    special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s.,;:'\"()\-]")
    multiple_spaces_pattern = re.compile(r"\s+")

    data = {
        "wiki_description": None,
        "wiki_course": None,
        "wiki_origin": None,
        "wiki_serving_temp": None,
        "wiki_categories": [],
        "is_redirect": False,
    }

    if redirect_pattern.search(article_xml):
        data["is_redirect"] = True
        return data

    description_section_pattern = re.compile(
        r"={2,}\s*Description\s*={2,}(.*?)(?=\n={2,}|$)", re.IGNORECASE | re.DOTALL
    )
    match_desc = description_section_pattern.search(article_xml)

    description = None
    if match_desc:
        description = match_desc.group(1)

    if not description:
        match = lead_section_pattern.search(article_xml)
        if match:
            description = match.group(1)
        else:
            description = article_xml[:2000]

    description = refs_pattern.sub("", description)
    description = templates_pattern.sub("", description)
    description = wiki_links_pattern.sub(r"\1", description)
    description = bold_italic_pattern.sub(r"\1", description)
    description = html_tags_pattern.sub("", description)
    description = special_chars_pattern.sub(" ", description)
    description = multiple_spaces_pattern.sub(" ", description)

    paragraphs = [p.strip() for p in description.split("\n") if len(p.strip()) > 50]
    if paragraphs:
        data["wiki_description"] = paragraphs[0]

    infobox_match = re.search(r"{{Infobox\s+([^\n]+)", article_xml, re.IGNORECASE)
    if infobox_match:
        infobox_context = article_xml[:5000]

        def extract_field(field_names):
            for name in field_names:
                # Improved regex: look for value until next line starting with | or }}
                pattern = re.compile(
                    r"\|\s*" + name + r"\s*=\s*(.*?)(?=\n\s*\||\n\s*}}|$)",
                    re.IGNORECASE | re.DOTALL,
                )
                match = pattern.search(infobox_context)
                if match:
                    val = match.group(1).strip()
                    val = refs_pattern.sub("", val)
                    val = templates_pattern.sub("", val)
                    val = wiki_links_pattern.sub(r"\1", val)
                    val = html_tags_pattern.sub("", val)
                    val = multiple_spaces_pattern.sub(" ", val).strip()
                    if val:
                        return val
            return None

        data["wiki_course"] = extract_field(["course"])
        data["wiki_origin"] = extract_field(["place_of_origin", "origin", "country"])
        data["wiki_serving_temp"] = extract_field(
            ["serving_temperature", "temperature"]
        )

    categories = re.findall(
        r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", article_xml, re.IGNORECASE
    )
    if categories:
        data["wiki_categories"] = [c.strip() for c in categories]

    return data


def process_partition(iterator, dump_path_str):
    import bz2
    import re
    from pathlib import Path

    rows = list(iterator)
    if not rows:
        return []

    # Sort by offset to minimize disk seeking
    rows.sort(key=lambda x: x.wiki_offset)

    dump_path = Path(dump_path_str)

    results = []

    try:
        with open(dump_path, "rb") as raw_file:
            for row in rows:
                offset = row.wiki_offset
                try:
                    raw_file.seek(offset)

                    decompressor = bz2.BZ2Decompressor()
                    article_xml = None

                    target_page_id = row.page_id

                    article_lines = []

                    while True:
                        chunk = raw_file.read(8192)
                        if not chunk:
                            break

                        try:
                            decompressed = decompressor.decompress(chunk)
                            if decompressed:
                                text = decompressed.decode("utf-8", errors="ignore")
                                article_lines.append(text)

                                full_text = "".join(article_lines)

                                if "</page>" in full_text:
                                    parts = full_text.split("</page>")

                                    found = False
                                    for i in range(len(parts) - 1):
                                        page_xml = parts[i] + "</page>"

                                        current_id_match = re.search(
                                            r"<id>(\d+)</id>", page_xml
                                        )
                                        if current_id_match:
                                            current_id = int(current_id_match.group(1))
                                            if current_id == target_page_id:
                                                article_xml = page_xml
                                                found = True
                                                break

                                    if found:
                                        break

                                    article_lines = [parts[-1]]

                        except EOFError:
                            break

                    if article_xml:
                        data = extract_wiki_data_from_xml(article_xml)

                        results.append(
                            (
                                offset,
                                data["wiki_description"],
                                data["wiki_course"],
                                data["wiki_origin"],
                                data["wiki_serving_temp"],
                                data["wiki_categories"],
                                data["is_redirect"],
                            )
                        )
                    else:
                        results.append((offset, None, None, None, None, [], False))

                except Exception:
                    results.append((offset, None, None, None, None, [], False))

    except Exception:
        pass

    return results


# Enrich matches with descriptions using Spark parallel processing
def enrich_with_descriptions(spark, matches_df, dump_path: Path, logger):
    logger.info("=" * 70)
    logger.info("EXTRACTING WIKIPEDIA DESCRIPTIONS")
    logger.info("=" * 70)
    logger.info(f"Dump file: {dump_path}")

    if not dump_path.exists():
        logger.warning(f"Wikipedia dump not found at {dump_path}")
        logger.warning("Skipping description extraction")
        return matches_df

    matches_with_offset = matches_df.filter(F.col("wiki_offset").isNotNull())

    unique_offsets = matches_with_offset.select("wiki_offset", "page_id").distinct()

    # Repartition to ensure parallelism.
    unique_offsets = unique_offsets.repartition(20)

    logger.info("Processing offsets in parallel...")

    dump_path_str = str(dump_path)

    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    result_schema = StructType(
        [
            StructField("wiki_offset", LongType(), True),
            StructField("wiki_description", StringType(), True),
            StructField("wiki_course", StringType(), True),
            StructField("wiki_origin", StringType(), True),
            StructField("wiki_serving_temp", StringType(), True),
            StructField("wiki_categories", ArrayType(StringType()), True),
            StructField("is_redirect", BooleanType(), True),
        ]
    )

    descriptions_rdd = unique_offsets.rdd.mapPartitions(
        lambda iterator: process_partition(iterator, dump_path_str)
    )

    descriptions_df = spark.createDataFrame(descriptions_rdd, schema=result_schema)

    logger.info("Joining extracted descriptions back to matches...")

    if "wiki_description" in matches_df.columns:
        matches_df = matches_df.drop("wiki_description")

    enriched_matches = matches_df.join(descriptions_df, "wiki_offset", "left")

    logger.info("Selecting best non-redirect match per recipe...")

    from pyspark.sql.window import Window

    # Logic:
    # 1. Prefer is_redirect = False
    # 2. Then prefer rank (lower is better)
    window_spec = Window.partitionBy("recipe_url").orderBy(
        F.col("is_redirect").asc(),
        F.col("rank").asc(),
    )

    best_matches = (
        enriched_matches.withColumn("selection_rank", F.row_number().over(window_spec))
        .filter(F.col("selection_rank") == 1)
        .drop("selection_rank", "is_redirect", "rank", "similarity")
    )

    return best_matches


# Save Spark DataFrame as single JSON file
def save_dataframe_single_file(df, output_path: Path, logger):
    temp_path = str(output_path) + "_temp"
    logger.info(f"Saving to temporary path: {temp_path}")

    df.coalesce(1).write.mode("overwrite").option("ignoreNullFields", "false").json(
        temp_path
    )

    part_files = glob.glob(f"{temp_path}/part-*.json")
    if not part_files:
        raise RuntimeError(f"No part file found in {temp_path}")

    source_file = part_files[0]

    if output_path.exists():
        if output_path.is_dir():
            logger.warning(f"Removing existing directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    logger.info(f"Moving {source_file} to {output_path}")
    shutil.move(source_file, str(output_path))

    shutil.rmtree(temp_path)
    logger.info(f"Successfully saved to {output_path}")


def main():
    logger = config.setup_logging(config.WIKI_SPARK_LOG)

    logger.info("=" * 70)
    logger.info("WIKIPEDIA RECIPE MATCHING")
    logger.info("=" * 70)
    logger.info("")

    logger.info("Initializing Spark session...")
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Wikipedia Recipe Matcher")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        recipes_file = config.RECIPES_FILE
        if not Path(recipes_file).exists():
            logger.error(f"Recipes file not found: {recipes_file}")
            return

        logger.info(f"Loading recipes from: {recipes_file}")

        recipes_df = spark.read.json(str(recipes_file))

        recipes_df = recipes_df.filter(
            (F.col("title").isNotNull()) & (F.col("url").isNotNull())
        )

        recipe_count = recipes_df.count()
        logger.info(f"Loaded {recipe_count:,} recipes")

        index_path = Path(config.WIKI_INDEX_PATH)
        if not index_path.exists():
            logger.error(f"Wikipedia index not found: {index_path}")
            return

        wiki_df = load_wikipedia_index(spark, index_path, logger)
        logger.info("")

        matches_df = find_wikipedia_matches_spark(spark, recipes_df, wiki_df, logger)

        matches_output = Path(config.WIKI_RECIPES_OUTPUT)
        logger.info(f"Saving matches without descriptions to: {matches_output}")

        matches_no_desc = matches_df.drop("wiki_description")

        save_dataframe_single_file(matches_no_desc, matches_output, logger)

        match_count = matches_df.count()
        logger.info(f"Saved {match_count:,} matches without descriptions")
        logger.info("")

        dump_path = Path(config.WIKI_DUMP_PATH)
        enriched_df = enrich_with_descriptions(spark, matches_df, dump_path, logger)

        if "wiki_offset" in enriched_df.columns:
            enriched_df = enriched_df.drop("wiki_offset")

        save_dataframe_single_file(enriched_df, matches_output, logger)

    finally:
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Done.")


if __name__ == "__main__":
    main()
