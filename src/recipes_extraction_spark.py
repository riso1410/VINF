import os
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

import config


def create_extraction_udf():
    recipe_schema = StructType([
        StructField("url", StringType(), False),
        StructField("title", StringType(), False),
        StructField("description", StringType(), True),
        StructField("ingredients", ArrayType(StringType()), False),
        StructField("method", StringType(), True),
        StructField("chef", StringType(), True),
        StructField("difficulty", StringType(), True),
        StructField("prep_time", StringType(), True),
        StructField("servings", IntegerType(), True),
    ])

    @udf(returnType=recipe_schema)
    def extract_recipe_udf(path: str, content: bytes):
        try:
            from markitdown import MarkItDown
            from recipe_parser import parse_recipe_html, should_skip_metadata, metadata_to_dict

            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.html', delete=False) as f:
                    f.write(content)
                    temp_file = f.name

                converter = MarkItDown()
                metadata = parse_recipe_html(
                    temp_file,
                    markdown_converter=converter,
                    logger=None,
                )

                if metadata and not should_skip_metadata(metadata):
                    return metadata_to_dict(metadata)
                return None
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception:
            return None

    return extract_recipe_udf


def main():
    os.makedirs(config.SCRAPED_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    logger = config.setup_logging(config.RECIPES_SPARK_LOG)

    html_dir = config.RAW_HTML_DIR
    if not os.path.exists(html_dir):
        logger.error("HTML directory not found: %s", html_dir)
        return

    html_pattern = os.path.join(html_dir, "*.html")

    logger.info("Initializing Spark with %d partitions...", config.RECIPES_SPARK_PARTITIONS)

    spark = (
        SparkSession.builder
        .appName("RecipesExtractionSpark")
        .master("local[*]")
        .config("spark.default.parallelism", str(config.RECIPES_SPARK_PARTITIONS))
        .config("spark.sql.shuffle.partitions", str(config.RECIPES_SPARK_PARTITIONS))
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .getOrCreate()
    )

    try:
        logger.info("Reading HTML files using binaryFile source...")
        html_df = spark.read.format("binaryFile").load(html_pattern)

        total_files = html_df.count()
        logger.info("Found %d HTML files", total_files)

        if total_files == 0:
            logger.error("No HTML files found matching: %s", html_pattern)
            return

        logger.info("Extracting recipes using UDF-based processing...")
        extract_udf = create_extraction_udf()

        recipes_df = (
            html_df
            .repartition(config.RECIPES_SPARK_PARTITIONS)
            .withColumn("recipe", extract_udf(col("path"), col("content")))
            .filter(col("recipe").isNotNull())
            .select("recipe.*")
            .orderBy("url")
            .cache()
        )

        success_count = recipes_df.count()
        skipped_count = total_files - success_count

        logger.info(
            "Extraction finished: %d recipes extracted, %d skipped or failed",
            success_count,
            skipped_count,
        )

        logger.info("Writing recipes to JSONL...")
        temp_output = config.RECIPES_FILE + "_temp"

        recipes_df.coalesce(1).write.mode("overwrite").json(temp_output)

        import glob
        import shutil
        part_files = glob.glob(os.path.join(temp_output, "part-*.json"))
        if part_files:
            shutil.move(part_files[0], config.RECIPES_FILE)
            shutil.rmtree(temp_output)

        logger.info("✅ Successfully wrote %d recipes to %s", success_count, config.RECIPES_FILE)

        recipes_df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
