import json
import os

from pathlib import Path
from typing import Iterator
from pyspark.sql import SparkSession

import config
from recipe_parser import metadata_to_dict, parse_recipe_html, should_skip_metadata


def process_partition(html_paths: Iterator[str]) -> Iterator[object]:
    from markitdown import MarkItDown

    converter = MarkItDown()
    for html_path in html_paths:
        metadata = parse_recipe_html(
            html_path,
            markdown_converter=converter,
            logger=None,
        )
        if metadata and not should_skip_metadata(metadata):
            yield metadata


def determine_partitions(
    file_count: int,
    *,
    files_per_partition: int,
) -> int:
    if file_count == 0:
        return 0

    base = max(files_per_partition, 1)
    estimated = max(1, file_count // base)
    return min(max(estimated, 1), file_count)


def write_jsonl(output_path: str, metadata_items: list[object]) -> None:
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8") as fp:
        for metadata in metadata_items:
            json.dump(metadata_to_dict(metadata), fp, ensure_ascii=False)
            fp.write("\n")


def ship_dependency_modules(spark_context) -> None:
    """
    Ensure local modules used inside mapPartitions are available on executors.
    """
    base_dir = Path(__file__).resolve().parent
    for module_name in ("recipe_parser.py",):
        module_path = base_dir / module_name
        if module_path.exists():
            spark_context.addPyFile(str(module_path))


def main():
    os.makedirs(config.SCRAPED_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    logger = config.setup_logging(config.RECIPES_SPARK_LOG)

    html_dir = config.RAW_HTML_DIR
    if not os.path.exists(html_dir):
        logger.error("HTML directory not found: %s", html_dir)
        return

    html_files = [
        os.path.join(html_dir, name)
        for name in os.listdir(html_dir)
        if name.endswith(".html")
    ]

    total_files = len(html_files)
    if total_files == 0:
        logger.error("No HTML files found under %s", html_dir)
        return

    partitions = determine_partitions(
        total_files,
        files_per_partition=config.RECIPES_SPARK_FILES_PER_PARTITION,
    )

    logger.info(
        "Starting Spark extraction for %d files using %d partitions",
        total_files,
        partitions,
    )

    builder = SparkSession.builder.appName("RecipesExtractionSpark")

    spark = builder.getOrCreate()
    try:
        sc = spark.sparkContext
        ship_dependency_modules(sc)

        metadata = (
            sc.parallelize(html_files, partitions)
            .mapPartitions(process_partition)
            .collect()
        )

        metadata.sort(key=lambda item: item.url)

        success_count = len(metadata)
        skipped_count = total_files - success_count

        logger.info(
            "Extraction finished: %d recipes extracted, %d skipped or failed",
            success_count,
            skipped_count,
        )

        write_jsonl(config.RECIPES_FILE, metadata)
        logger.info("Wrote recipes to %s", config.RECIPES_FILE)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
