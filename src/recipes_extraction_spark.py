import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import col, size

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.RECIPES_SPARK_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_html_file(html_file_path: str):
    # Import inside function to avoid pickle issues with Spark
    from markitdown import MarkItDown
    from recipe_parser import parse_recipe_html, should_skip_metadata

    try:
        if not Path(html_file_path).exists():
            return None

        # Convert HTML to markdown
        converter = MarkItDown()
        metadata = parse_recipe_html(
            html_file_path,
            markdown_converter=converter,
            logger=None,
        )

        # Skip if invalid or should be skipped
        if not metadata or should_skip_metadata(metadata):
            return None

        # Convert to dictionary and return
        recipe_data = {
            'url': metadata.url or '',
            'title': metadata.title or '',
            'description': metadata.description or '',
            'ingredients': metadata.ingredients or [],
            'method': metadata.method or '',
            'chef': metadata.chef or '',
            'difficulty': metadata.difficulty or '',
            'prep_time': metadata.prep_time or '',
            'servings': metadata.servings or '',
        }

        return recipe_data

    except Exception:
        # Silent failures in workers - return None
        return None


def process_with_spark(
    html_dir: str = None,
    output_file: str = None,
    num_partitions: int = None
):
    """
    Process HTML files using PySpark for distributed processing.

    Args:
        html_dir: Directory containing HTML files
        output_file: Output file path for recipes
        num_partitions: Number of Spark partitions for parallel processing
    """
    # Use config defaults if not specified
    html_dir = html_dir or config.RAW_HTML_DIR
    output_file = output_file or config.RECIPES_FILE
    num_partitions = num_partitions or config.RECIPES_SPARK_PARTITIONS

    logger.info("=" * 80)
    logger.info("RECIPE EXTRACTION WITH PYSPARK")
    logger.info("=" * 80)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Recipe Extractor") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", str(num_partitions)) \
        .getOrCreate()

    logger.info(f"Spark session created: {spark.sparkContext.applicationId}")

    # Ship recipe_parser module to executors
    recipe_parser_path = Path(__file__).parent / "recipe_parser.py"
    if recipe_parser_path.exists():
        spark.sparkContext.addPyFile(str(recipe_parser_path))
        logger.info("Shipped recipe_parser.py to Spark executors")

    try:
        # Get all HTML files
        html_path = Path(html_dir)
        if not html_path.exists():
            logger.error(f"Directory not found: {html_dir}")
            return

        html_files = list(html_path.glob("*.html"))
        html_file_paths = [str(f) for f in html_files]

        if not html_file_paths:
            logger.error("No HTML files found")
            return

        logger.info(f"Found {len(html_file_paths):,} HTML files to process")

        # Create RDD from file paths
        file_rdd = spark.sparkContext.parallelize(html_file_paths, num_partitions)

        # Process files in parallel
        logger.info("Processing files with Spark...")
        results_rdd = file_rdd.map(process_html_file)

        # Filter out None results (skipped or failed files)
        valid_results = results_rdd.filter(lambda x: x is not None)

        # Collect results
        results = valid_results.collect()

        logger.info(f"Successfully processed {len(results):,} out of {len(html_file_paths):,} files")

        # Define schema for DataFrame
        schema = StructType([
            StructField("url", StringType(), True),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("ingredients", ArrayType(StringType()), True),
            StructField("method", StringType(), True),
            StructField("chef", StringType(), True),
            StructField("difficulty", StringType(), True),
            StructField("prep_time", StringType(), True),
            StructField("servings", StringType(), True),
        ])

        # Create DataFrame
        df = spark.createDataFrame(results, schema=schema)

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output
        logger.info(f"Writing output to {output_file}")

        # Write single JSONL file (main output)
        df.coalesce(1).write.mode("overwrite").json(str(output_path) + ".tmp")

        # Move from Spark output directory to final location
        import shutil
        tmp_dir = Path(str(output_path) + ".tmp")
        part_files = list(tmp_dir.glob("part-*.json"))

        if part_files:
            shutil.move(str(part_files[0]), output_file)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Written JSONL to: {output_file}")
        else:
            logger.error("No output files generated!")

        # Print statistics
        logger.info("=" * 80)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total files processed: {len(html_file_paths):,}")
        logger.info(f"Successfully extracted: {len(results):,}")
        logger.info(f"Skipped/Failed: {len(html_file_paths) - len(results):,}")

        # Show sample of data
        logger.info("\nSample of extracted recipes:")
        df.show(5, truncate=True)

        # Show field completeness
        logger.info("\nField completeness:")
        for field in df.columns:
            if field != 'url':  # URL is always present
                # Special handling for arrays (ingredients)
                if field == 'ingredients':
                    count = df.filter(
                        (col(field).isNotNull()) & (size(col(field)) > 0)
                    ).count()
                else:
                    count = df.filter(col(field).isNotNull()).count()

                percentage = (count / len(results)) * 100 if results else 0
                logger.info(f"  {field}: {count:,}/{len(results):,} ({percentage:.1f}%)")

        # Show top chefs
        if df.filter(col('chef').isNotNull()).count() > 0:
            logger.info("\nTop 5 chefs by recipe count:")
            chef_counts = df.groupBy('chef').count().orderBy(col('count').desc()).limit(5)
            for row in chef_counts.collect():
                if row['chef']:
                    logger.info(f"  {row['chef']}: {row['count']:,} recipes")

        # Show difficulty distribution
        if df.filter(col('difficulty').isNotNull()).count() > 0:
            logger.info("\nRecipes by difficulty:")
            difficulty_counts = df.groupBy('difficulty').count().orderBy(col('count').desc())
            for row in difficulty_counts.collect():
                if row['difficulty']:
                    logger.info(f"  {row['difficulty']}: {row['count']:,} recipes")

        logger.info("=" * 80)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during Spark processing: {str(e)}", exc_info=True)
        raise
    finally:
        # Stop Spark session
        spark.stop()
        logger.info("Spark session stopped")


def main():
    """Main entry point for the recipe extractor."""
    process_with_spark(
        html_dir=config.RAW_HTML_DIR,
        output_file=config.RECIPES_FILE,
        num_partitions=config.RECIPES_SPARK_PARTITIONS
    )


if __name__ == "__main__":
    main()
