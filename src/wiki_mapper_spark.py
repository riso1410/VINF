import bz2
import glob
import json
import os
import re
import shutil
import sys
from typing import Iterable, Iterator

import pyspark.sql.functions as F
import pyspark.sql.types as T
from fuzzywuzzy import fuzz
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.column import Column

import config

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def sanitize_ingredient_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", "", value.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def sanitize_ingredient_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    sanitized: list[str] = []
    for entry in values:
        token = sanitize_ingredient_text(entry)
        if token:
            sanitized.append(token)
    return sanitized


def extract_ingredients_from_wikitext(wikitext: str) -> list[str]:
    if not wikitext:
        return []
    ingredients = set()
    infobox_pattern = r"\{\{[Ii]nfobox\s+[Ff]ood.*?\}\}"
    for infobox in re.findall(infobox_pattern, wikitext, re.DOTALL):
        fields = re.findall(r"\|\s*(?:main_)?ingredient[s]?\s*=\s*([^\|]+)", infobox, re.IGNORECASE)
        for field in fields:
            text = re.sub(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", r"\1", field)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\{\{[^}]+\}\}", "", text)
            for part in re.split(r"[,;\n]", text):
                token = sanitize_ingredient_text(part)
                if len(token) > 2:
                    ingredients.add(token)
    for item in re.findall(r"[\*#]\s*([^\n]+)", wikitext)[:20]:
        text = re.sub(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", r"\1", item)
        text = sanitize_ingredient_text(re.sub(r"<[^>]+>", "", text))
        if 2 < len(text) < 50:
            ingredients.add(text)
    return list(ingredients)[:50]


def is_food_article(title: str, wikitext: str, categories: str) -> bool:
    if not title or not wikitext:
        return False
    lowered = title.lower()
    if any(tag in lowered for tag in ("disambiguation", "list of", "category:", "template:", "file:", "user:", "talk:")):
        return False
    if re.search(r"\{\{[Ii]nfobox\s+[Ff]ood", wikitext):
        return True
    return bool(categories) and "food" in categories.lower()


def title_similarity(recipe_title: str, wiki_title: str) -> float:
    if not recipe_title or not wiki_title:
        return 0.0
    return fuzz.token_sort_ratio(recipe_title, wiki_title) / 100.0


def ingredient_similarity(recipe_ingredients: list[str], wiki_ingredients: list[str]) -> float:
    if not recipe_ingredients or not wiki_ingredients:
        return 0.0
    recipe_set = {ing for ing in recipe_ingredients if ing}
    wiki_set = {ing for ing in wiki_ingredients if ing}
    if not recipe_set or not wiki_set:
        return 0.0
    return len(recipe_set & wiki_set) / len(recipe_set | wiki_set)


def build_wiki_url(title: str | None) -> str:
    if not title:
        return ""
    sanitized = re.sub(r"\s+", "_", title.strip())
    return f"https://en.wikipedia.org/wiki/{sanitized}"


def extract_between_tags(page: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", page, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    content = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", match.group(1), flags=re.DOTALL)
    return content.strip()


def consolidate_batches(temp_root: str, output_file: str, logger) -> int:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    batch_dirs = sorted(glob.glob(os.path.join(temp_root, "batch_*")))
    if not batch_dirs:
        open(output_file, "w", encoding="utf-8").close()
        shutil.rmtree(temp_root, ignore_errors=True)
        return 0
    if os.path.exists(output_file):
        os.remove(output_file)
    total = 0
    with open(output_file, "w", encoding="utf-8") as destination:
        for batch in batch_dirs:
            for part in sorted(glob.glob(os.path.join(batch, "part-*.json"))):
                with open(part, "r", encoding="utf-8") as source:
                    for line in source:
                        line = line.strip()
                        if line:
                            destination.write(line + "\n")
                            total += 1
    shutil.rmtree(temp_root, ignore_errors=True)
    return total


def write_dataframe_to_jsonl(df: DataFrame, output_file: str, temp_root: str, logger) -> int:
    shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(temp_root, exist_ok=True)
    df.write.mode("overwrite").json(os.path.join(temp_root, "batch_0"))
    return consolidate_batches(temp_root, output_file, logger)


EXTRACT_INGREDIENTS_UDF = F.udf(extract_ingredients_from_wikitext, T.ArrayType(T.StringType()))
IS_FOOD_ARTICLE_UDF = F.udf(is_food_article, T.BooleanType())
TITLE_SIMILARITY_UDF = F.udf(title_similarity, T.FloatType())
INGREDIENT_SIMILARITY_UDF = F.udf(ingredient_similarity, T.FloatType())
SANITIZE_INGREDIENTS_UDF = F.udf(sanitize_ingredient_list, T.ArrayType(T.StringType()))
WIKI_URL_UDF = F.udf(build_wiki_url, T.StringType())

WIKI_SCHEMA = T.StructType(
    [
        T.StructField("wiki_title", T.StringType(), nullable=False),
        T.StructField("wiki_text", T.StringType(), nullable=False),
        T.StructField("redirect_title", T.StringType(), nullable=True),
    ]
)

MATCH_SCHEMA = T.StructType(
    [
        T.StructField("doc_id", T.StringType(), nullable=False),
        T.StructField("recipe_title", T.StringType(), nullable=True),
        T.StructField("wiki_title", T.StringType(), nullable=True),
        T.StructField("wiki_url", T.StringType(), nullable=True),
        T.StructField("wiki_ingredients", T.ArrayType(T.StringType()), nullable=True),
        T.StructField("match_score", T.DoubleType(), nullable=True),
    ]
)

def create_spark_session() -> SparkSession:
    return SparkSession.builder.appName("WikipediaRecipeMapper").master("local[*]").getOrCreate()


class WikipediaRecipeMapper:
    def __init__(
        self,
        spark: SparkSession,
        wiki_dump_path: str,
        output_path: str,
        recipes_path: str | None = None,
    ) -> None:
        self.spark = spark
        self.wiki_dump_path = wiki_dump_path
        self.output_path = output_path
        self.recipes_path = recipes_path or config.RECIPES_FILE
        self.logger = config.setup_logging(config.WIKI_SPARK_LOG)
        self.batch_size = config.WIKI_STREAM_BATCH_SIZE
        self.clean_checkpoints = config.WIKI_CLEAN_CHECKPOINTS
        self.checkpoint_dir = config.WIKI_CHECKPOINT_DIR
        self.checkpoint_file = config.WIKI_CHECKPOINT_FILE
        self.checkpoint_flush_batches = max(1, config.WIKI_CHECKPOINT_FLUSH_BATCHES)
        self.batches_since_checkpoint = 0
        self.file_progress: dict[str, int] = {}
        self.best_matches: dict[str, dict[str, object]] = {}
        self.example_logged = False
        self.checkpoint_completed = False
        self.recipes_df: DataFrame | None = None

    def run(self) -> None:
        self.prepare_environment()
        self.load_checkpoint_state()
        self.load_recipes()
        matches_df = self.process_wiki_stream()
        final_df = self.join_matches_with_recipes(matches_df)
        scratch_dir = os.path.join(config.DATA_DIR, "wiki_mapper_temp")
        total = write_dataframe_to_jsonl(final_df, self.output_path, scratch_dir, self.logger)
        self.logger.info("Final mapping written to %s with %s recipes", self.output_path, total)
        self.save_checkpoint_state(completed=True)

    def prepare_environment(self) -> None:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        if self.clean_checkpoints and os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            self.logger.info("Removed existing wiki-mapper checkpoints at %s", self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint_state(self) -> None:
        if not os.path.exists(self.checkpoint_file):
            self.logger.info("No existing wiki-mapper checkpoint found; starting fresh")
            return
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to load wiki-mapper checkpoint %s: %s", self.checkpoint_file, exc)
            return
        self.best_matches = state.get("best_matches", {})
        raw_progress = state.get("file_progress", {}) or {}
        self.file_progress = {str(path): int(value) for path, value in raw_progress.items()}
        self.checkpoint_completed = bool(state.get("completed", False))
        self.logger.info(
            "Loaded wiki-mapper checkpoint: %s matches, %s files resumed",
            len(self.best_matches),
            len(self.file_progress),
        )

    def save_checkpoint_state(self, completed: bool | None = None) -> None:
        if completed is not None:
            self.checkpoint_completed = completed
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        state = {
            "best_matches": self.best_matches,
            "file_progress": self.file_progress,
            "completed": self.checkpoint_completed,
        }
        temp_path = self.checkpoint_file + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle)
            os.replace(temp_path, self.checkpoint_file)
        except OSError as exc:
            self.logger.warning("Unable to write wiki-mapper checkpoint %s: %s", self.checkpoint_file, exc)

    def flush_checkpoint_if_needed(self) -> None:
        self.batches_since_checkpoint += 1
        if self.batches_since_checkpoint < self.checkpoint_flush_batches:
            return
        self.save_checkpoint_state()
        self.batches_since_checkpoint = 0

    def load_recipes(self) -> None:
        df = self.spark.read.json(self.recipes_path)

        df = df.withColumn(
            "doc_id",
            F.sha2(F.coalesce(F.col("url"), F.col("title")), 256),
        )

        string_columns = [name for name, dtype in df.dtypes if dtype == "string"]
        if string_columns:
            df = df.fillna("", subset=string_columns)

        df = df.withColumn("ingredients", SANITIZE_INGREDIENTS_UDF(F.col("ingredients")))

        self.recipes_df = df.persist()
        self.logger.info("Loaded %s recipes", self.recipes_df.count())

    def process_wiki_stream(self) -> DataFrame | None:
        batch_index = 0
        for batch_df, count in self.yield_wiki_batches():
            batch_index += 1
            self.logger.info("Processing Wikipedia batch %s (%s pages)", batch_index, count)
            matches = self.compute_matches(batch_df)
            if matches is None:
                self.flush_checkpoint_if_needed()
                continue
            rows = matches.collect()
            if not rows:
                self.flush_checkpoint_if_needed()
                continue
            for row in rows:
                self.update_best_match(row.asDict(recursive=True), batch_index)
            self.flush_checkpoint_if_needed()
        if self.batches_since_checkpoint:
            self.save_checkpoint_state()
            self.batches_since_checkpoint = 0
        if not self.best_matches:
            return None
        return self.spark.createDataFrame(list(self.best_matches.values()), MATCH_SCHEMA)

    def compute_matches(self, wiki_df: DataFrame) -> DataFrame | None:
        recipes = self.recipes_df
        if recipes is None:
            return None
        enriched = (
            wiki_df
            .withColumn("categories", F.regexp_extract(F.col("wiki_text"), r"\[\[Category:([^\]]+)\]\]", 1))
            .filter(IS_FOOD_ARTICLE_UDF(F.col("wiki_title"), F.col("wiki_text"), F.col("categories")))
            .withColumn("wiki_ingredients", SANITIZE_INGREDIENTS_UDF(EXTRACT_INGREDIENTS_UDF(F.col("wiki_text"))))
            .withColumn(
                "resolved_title",
                F.when(F.length(F.col("redirect_title")) > 0, F.col("redirect_title")).otherwise(F.col("wiki_title")),
            )
            .withColumn("wiki_url", WIKI_URL_UDF(F.col("resolved_title")))
            .withColumn("wiki_title", F.col("resolved_title"))
            .select("wiki_title", "wiki_url", "wiki_ingredients")
        )
        if not self.example_logged:
            sample_rows = enriched.select("wiki_title", "wiki_url").limit(1).collect()
            if sample_rows:
                sample_row = sample_rows[0]
                self.logger.info("Example Wikipedia page: %s (%s)", sample_row["wiki_title"], sample_row["wiki_url"])
                self.example_logged = True
        if enriched.rdd.isEmpty():
            return None
        scored = (
            F.broadcast(recipes)
            .crossJoin(enriched)
            .withColumn("title_similarity", TITLE_SIMILARITY_UDF(F.col("title").alias("recipe_title"), F.col("wiki_title")))
        )
        scored = scored.withColumn(
            "ingredient_similarity",
            INGREDIENT_SIMILARITY_UDF(F.col("ingredients"), F.col("wiki_ingredients")),
        )
        scored = scored.withColumn(
            "match_score",
            F.col("title_similarity") * F.lit(0.7) + F.col("ingredient_similarity") * F.lit(0.3),
        )
        scored = scored.filter(F.col("match_score") >= F.lit(0.0))

        ranked = scored.select(
            F.col("doc_id"),
            F.col("title").alias("recipe_title"),
            F.col("wiki_title"),
            F.col("wiki_url"),
            F.col("wiki_ingredients"),
            F.col("match_score"),
        )
        window = Window.partitionBy("doc_id").orderBy(F.desc("match_score"))
        return ranked.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") == 1).drop("rank")

    def yield_wiki_batches(self) -> Iterator[tuple[DataFrame, int]]:
        records: list[tuple[str, str, str]] = []
        for page_xml, source_path, page_index in self.stream_wiki_pages():
            parsed = self.page_to_tuple(page_xml)
            if not parsed:
                continue
            title, text, redirect_title = parsed
            records.append((title, text, redirect_title))
            self.file_progress[source_path] = page_index
            if len(records) >= self.batch_size:
                yield self.spark.createDataFrame(records, schema=WIKI_SCHEMA), len(records)
                records = []
        if records:
            yield self.spark.createDataFrame(records, schema=WIKI_SCHEMA), len(records)

    def stream_wiki_pages(self) -> Iterator[tuple[str, str, int]]:
        if os.path.isdir(self.wiki_dump_path):
            targets = sorted(glob.glob(os.path.join(self.wiki_dump_path, "*.xml*")))
        else:
            targets = [self.wiki_dump_path]
        for path in targets:
            opener = bz2.open if path.lower().endswith(".bz2") else open
            processed_pages = int(self.file_progress.get(path, 0))
            current_index = 0
            if processed_pages:
                self.logger.info(
                    "Resuming %s at page %s",
                    os.path.basename(path),
                    processed_pages + 1,
                )
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for page_xml in self.extract_pages(handle):
                    current_index += 1
                    if current_index <= processed_pages:
                        continue
                    yield page_xml, path, current_index
            self.file_progress[path] = current_index

    @staticmethod
    def extract_pages(stream: Iterable[str]) -> Iterator[str]:
        buffer: list[str] = []
        inside = False
        for line in stream:
            if "<page" in line:
                inside = True
                buffer = []
            if inside:
                buffer.append(line)
            if inside and "</page>" in line:
                inside = False
                yield "".join(buffer)

    @staticmethod
    def page_to_tuple(page_xml: str) -> tuple[str, str, str] | None:
        title = extract_between_tags(page_xml, "title")
        text = extract_between_tags(page_xml, "text")
        redirect_match = re.search(r"<redirect[^>]*title=\"([^\"]+)\"", page_xml)
        redirect_title = redirect_match.group(1) if redirect_match else ""
        if not title or not text:
            return None
        return title, text, redirect_title

    def update_best_match(self, candidate: dict[str, object], batch_index: int) -> None:
        doc_id = candidate.get("doc_id")
        if not doc_id:
            return
        candidate_score = float(candidate.get("match_score") or 0.0)
        existing = self.best_matches.get(doc_id)
        if existing and float(existing.get("match_score") or 0.0) >= candidate_score:
            return
        self.best_matches[doc_id] = candidate
        self.logger.debug(
            "Batch %s mapping recipe %s to %s (score %.3f)",
            batch_index,
            doc_id,
            candidate.get("wiki_title"),
            candidate_score,
        )

    def join_matches_with_recipes(self, matches_df: DataFrame | None) -> DataFrame:
        if self.recipes_df is None:
            raise RuntimeError("Recipes DataFrame not prepared")
        recipes = self.recipes_df
        if matches_df is None:
            merged = recipes.withColumn("wiki_title", F.lit(""))
            merged = merged.withColumn("wiki_url", F.lit(""))
            merged = merged.withColumn("wiki_ingredients", F.expr("array()").cast(T.ArrayType(T.StringType())))
            merged = merged.withColumn("match_score", F.lit(0.0))
            return merged
        working = matches_df.drop("recipe_title") if "recipe_title" in matches_df.columns else matches_df
        joined = recipes.join(working, on="doc_id", how="left")
        return (
            joined.fillna({"wiki_title": "", "wiki_url": ""})
            .withColumn(
                "wiki_ingredients",
                F.when(F.col("wiki_ingredients").isNull(), F.expr("array()").cast(T.ArrayType(T.StringType()))).otherwise(F.col("wiki_ingredients")),
            )
            .withColumn(
                "match_score",
                F.when(F.col("match_score").isNull(), F.lit(0.0)).otherwise(F.col("match_score")),
            )
        )
def main() -> None:
    spark = create_spark_session()
    mapper = WikipediaRecipeMapper(
        spark,
        wiki_dump_path=config.WIKI_DUMP_PATH,
        output_path=config.WIKI_OUTPUT_PATH,
        recipes_path=config.RECIPES_FILE,
    )
    mapper.run()
    spark.stop()


if __name__ == "__main__":
    main()
