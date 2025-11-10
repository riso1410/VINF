import bz2
import json
import os
import re
import shutil

from pyspark.sql import SparkSession, Row
from pyspark import StorageLevel

import config


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


def has_food_infobox(wikitext: str) -> bool:
    if not wikitext:
        return False

    food_infobox_pattern = r"\{\{[Ii]nfobox\s+(?:[Ff]ood|[Dd]ish|[Bb]everage|[Dd]rink|[Cc]heese|[Bb]read|[Cc]ake|[Dd]essert|[Ss]oup|[Ss]alad|[Ss]auce|[Cc]ondiment|[Ii]ngredient|[Ss]pice|[Mm]eal|[Cc]uisine|[Pp]repared\s+[Ff]ood)"
    return bool(re.search(food_infobox_pattern, wikitext))


def extract_ingredients_from_wikitext(wikitext: str) -> list[str]:
    if not wikitext:
        return []
    ingredients = set()

    infobox_pattern = r"\{\{[Ii]nfobox\s+(?:[Ff]ood|[Dd]ish|[Bb]everage|[Dd]rink|[Cc]heese|[Bb]read|[Cc]ake|[Dd]essert|[Ss]oup|[Ss]alad|[Ss]auce|[Cc]ondiment|[Ii]ngredient|[Ss]pice|[Mm]eal|[Cc]uisine|[Pp]repared\s+[Ff]ood).*?\}\}"

    for infobox in re.findall(infobox_pattern, wikitext, re.DOTALL):
        fields = re.findall(
            r"\|\s*(?:main_)?ingredient[s]?\s*=\s*([^\|]+)", infobox, re.IGNORECASE
        )
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


def extract_description_from_wikitext(wikitext: str) -> str:
    """Extract description from Wikipedia wikitext - only the lead section."""
    if not wikitext:
        return ""

    # Extract only the lead section (before first ==)
    lead_match = re.search(r"^(.*?)(?:^==|\Z)", wikitext, re.MULTILINE | re.DOTALL)
    if not lead_match:
        return ""
    
    lead_text = lead_match.group(1).strip()
    if not lead_text:
        return ""
    
    # Clean wikitext markup
    lead_text = re.sub(r"\{\{[^}]+\}\}", "", lead_text)
    lead_text = re.sub(r"<ref[^>]*>.*?</ref>", "", lead_text, flags=re.DOTALL)
    lead_text = re.sub(r"<!--.*?-->", "", lead_text, flags=re.DOTALL)
    lead_text = re.sub(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", r"\1", lead_text)
    lead_text = re.sub(r"\[https?://[^\]]+\s+([^\]]+)\]", r"\1", lead_text)
    lead_text = re.sub(r"[\[\]']", "", lead_text)
    lead_text = re.sub(r"<[^>]+>", "", lead_text)
    lead_text = re.sub(r"\s+", " ", lead_text).strip()
    
    # Truncate to reasonable length
    if len(lead_text) > 500:
        sentences = re.split(r"[.!?]+\s+", lead_text[:600])
        result = []
        length = 0
        for sentence in sentences:
            if length + len(sentence) <= 500:
                result.append(sentence)
                length += len(sentence)
            else:
                break
        if result:
            lead_text = ". ".join(result) + "."
        else:
            lead_text = lead_text[:500] + "..."

    return lead_text


def normalize_title(title: str) -> str:
    if not title:
        return ""
    normalized = title.lower().strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def calculate_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio()


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


def process_recipe(recipe: dict) -> dict:
    import hashlib

    recipe["doc_id"] = hashlib.sha256(
        (recipe.get("url") or recipe.get("title", "")).encode()
    ).hexdigest()

    for key in [
        "url",
        "title",
        "description",
        "method",
        "chef",
        "difficulty",
        "prep_time",
        "servings",
    ]:
        recipe.setdefault(key, "")

    recipe["ingredients"] = sanitize_ingredient_list(recipe.get("ingredients"))
    return recipe


def extract_wiki_page(dump_path: str, offset_matches: tuple) -> list[dict]:
    offset, matches = offset_matches
    page_xml = read_page_at_offset(dump_path, offset)
    if not page_xml:
        return []

    namespace = extract_between_tags(page_xml, "ns")
    if namespace != "0":
        return []

    if "<redirect" in page_xml:
        return []

    title = extract_between_tags(page_xml, "title")
    text = extract_between_tags(page_xml, "text")
    if not text:
        return []

    if not has_food_infobox(text):
        return []

    wiki_data = {
        "wiki_title": title,
        "wiki_url": build_wiki_url(title),
        "wiki_ingredients": sanitize_ingredient_list(
            extract_ingredients_from_wikitext(text)
        ),
        "wiki_description": extract_description_from_wikitext(text),
    }

    return [
        {
            "doc_id": m["doc_id"],
            **wiki_data,
        }
        for m in matches
    ]


def merge_recipe_wiki(doc_id_recipe_wiki: tuple) -> dict:
    _, (recipe, wiki) = doc_id_recipe_wiki

    recipe.update(
        {
            "wiki_title": wiki.get("wiki_title", "") if wiki else "",
            "wiki_url": wiki.get("wiki_url", "") if wiki else "",
            "wiki_ingredients": wiki.get("wiki_ingredients", []) if wiki else [],
            "wiki_description": wiki.get("wiki_description", "") if wiki else "",
        }
    )
    return recipe


def remove_internal_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in ("doc_id", "offset", "similarity")}


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("WikipediaRecipeMapper")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.memory.fraction", "0.8")  
        .config("spark.memory.storageFraction", "0.5")  
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .getOrCreate()
    )


def parse_index_line(line: str) -> tuple[int, int, str] | None:
    if not line or not line.strip():
        return None

    parts = line.strip().split(":", 2)
    offset = int(parts[0])
    page_id = int(parts[1])
    title = parts[2]
    return (offset, page_id, title)


def read_page_at_offset(dump_path: str, offset: int) -> str | None:
    with open(dump_path, "rb") as f:
        f.seek(offset)
        decompressor = bz2.BZ2Decompressor()

        buffer = []
        found_page = False
        page_depth = 0

        # Read chunks
        while True:
            chunk = f.read(8192)  # 8KB chunks
            if not chunk:
                break

            try:
                text = decompressor.decompress(chunk).decode(
                    "utf-8", errors="ignore"
                )
            except EOFError:
                break

            buffer.append(text)

            for line in text.split("\n"):
                if "<page>" in line:
                    found_page = True
                    page_depth += 1
                if "</page>" in line:
                    page_depth -= 1
                    if found_page and page_depth == 0:
                        full_text = "".join(buffer)
                        match = re.search(r"<page>.*?</page>", full_text, re.DOTALL)
                        if match:
                            return match.group(0)
                        return None

    return None

class WikipediaRecipeMapper:
    def __init__(
        self,
        spark: SparkSession,
        wiki_dump_path: str,
        wiki_index_path: str,
        output_path: str,
        recipes_path: str | None = None,
    ) -> None:
        self.sc = spark.sparkContext
        self.wiki_dump_path = wiki_dump_path
        self.wiki_index_path = wiki_index_path
        self.output_path = output_path
        self.recipes_path = recipes_path or config.RECIPES_FILE
        self.logger = config.setup_logging(config.WIKI_SPARK_LOG)

    def run(self) -> None:
        self.prepare_environment()

        recipes_data = self.load_recipes()
        matches_data = self.match_titles(recipes_data)
        wiki_data_data = self.extract_wiki_data(matches_data)
        final_data = self.join_recipes_with_wiki_data(recipes_data, wiki_data_data)
        total = self.write_output(final_data)

        self.logger.info(
            "✓ Pipeline complete: %s recipes enriched with Wikipedia data", f"{total:,}"
        )

        recipes_data.unpersist()
        matches_data.unpersist()
        wiki_data_data.unpersist()

    def prepare_environment(self) -> None:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)

    def load_recipes(self):
        self.logger.info("STEP 1/5: Loading Recipes")

        rdd = (
            self.sc.textFile(self.recipes_path)
            .map(json.loads)
            .map(process_recipe)
            .persist(StorageLevel.MEMORY_ONLY)
        )

        self.logger.info("  Loaded %s recipes\n", f"{rdd.count():,}")
        return rdd

    def match_titles(self, recipes_data):
        self.logger.info("STEP 2/5: Matching Recipe Titles to Wikipedia")

        wiki_index_data = (
            self.sc.textFile(self.wiki_index_path)
            .map(parse_index_line)
            .filter(lambda x: x is not None)
        )

        total_wiki_pages = wiki_index_data.count()
        self.logger.info("  Loaded %s Wikipedia article pages", f"{total_wiki_pages:,}")
        
        recipes_with_normalized = recipes_data.map(
            lambda r: (normalize_title(r.get("title", "")), r)
        ).filter(lambda x: x[0])  
        
        wiki_with_normalized = wiki_index_data.map(
            lambda w: (normalize_title(w[2]), {"offset": w[0], "page_id": w[1], "wiki_title": w[2], "normalized": normalize_title(w[2])})
        ).filter(lambda x: x[0])  # Filter out empty titles
        
        # First try exact match with INNER JOIN
        exact_joined = recipes_with_normalized.join(wiki_with_normalized)
        
        exact_matches = exact_joined.map(
            lambda x: {
                "doc_id": x[1][0]["doc_id"],
                "wiki_title": x[1][1]["wiki_title"],
                "offset": x[1][1]["offset"],
                "similarity": 1.0,
            }
        )
        
        # Get recipes that didn't match exactly
        matched_doc_ids = exact_matches.map(lambda m: m["doc_id"]).collect()
        matched_doc_ids_set = set(matched_doc_ids)
        
        unmatched_recipes = recipes_data.filter(
            lambda r: r["doc_id"] not in matched_doc_ids_set
        )
        
        # Broadcast wiki index for fuzzy matching (only if we have unmatched recipes)
        unmatched_count = unmatched_recipes.count()
        self.logger.info(f"  Exact matches: {len(matched_doc_ids_set):,}")
        
        if unmatched_count > 0:
            self.logger.info(f"  Performing fuzzy matching for {unmatched_count:,} unmatched recipes...")
            
            # Collect wiki titles for fuzzy matching (this is expensive but necessary)
            wiki_titles = wiki_with_normalized.collect()
            wiki_titles_broadcast = self.sc.broadcast(wiki_titles)
            
            def find_best_fuzzy_match(recipe):
                recipe_title = normalize_title(recipe.get("title", ""))
                if not recipe_title:
                    return None
                
                best_match = None
                best_similarity = 0.5  # Minimum 50% similarity threshold
                
                for norm_wiki_title, wiki_data in wiki_titles_broadcast.value:
                    similarity = calculate_similarity(recipe_title, norm_wiki_title)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            "doc_id": recipe["doc_id"],
                            "wiki_title": wiki_data["wiki_title"],
                            "offset": wiki_data["offset"],
                            "similarity": similarity,
                        }
                
                return best_match
            
            fuzzy_matches = (
                unmatched_recipes
                .map(find_best_fuzzy_match)
                .filter(lambda x: x is not None)
            )
            
            # Combine exact and fuzzy matches
            matches_data = exact_matches.union(fuzzy_matches).persist(StorageLevel.MEMORY_ONLY)
            
            fuzzy_count = fuzzy_matches.count()
            self.logger.info(f"  Fuzzy matches (>50% similarity): {fuzzy_count:,}")
        else:
            matches_data = exact_matches.persist(StorageLevel.MEMORY_ONLY)

        matched_count = matches_data.count()
        total_recipes = recipes_data.count()

        self.logger.info(
            "Total matched: %s / %s recipes (%.1f%%)\n",
            f"{matched_count:,}",
            f"{total_recipes:,}",
            100.0 * matched_count / total_recipes if total_recipes > 0 else 0,
        )

        return matches_data

    def extract_wiki_data(self, matches_data):
        self.logger.info("STEP 3/5: Extracting Wikipedia Data")

        dump_path = self.wiki_dump_path

        from functools import partial
        extract_func = partial(extract_wiki_page, dump_path)

        wiki_data_data = (
            matches_data.map(lambda m: (m["offset"], m))
            .groupByKey()
            .flatMap(extract_func)
            .persist(StorageLevel.MEMORY_ONLY)
        )

        extracted_count = wiki_data_data.count()
        matched_pages = matches_data.count()
        self.logger.info(
            "Extracted data from %s pages (out of %s matched, %.1f%% had food infoboxes)\n",
            f"{extracted_count:,}",
            f"{matched_pages:,}",
            100.0 * extracted_count / matched_pages if matched_pages > 0 else 0,
        )

        return wiki_data_data

    def join_recipes_with_wiki_data(self, recipes_data, wiki_data_data):
        self.logger.info("STEP 4/5: Joining Recipes with Wikipedia Data")

        joined_data = (
            recipes_data.map(lambda r: (r["doc_id"], r))
            .leftOuterJoin(wiki_data_data.map(lambda w: (w["doc_id"], w)))
            .map(merge_recipe_wiki)
        )

        matched = joined_data.filter(lambda r: r["wiki_url"]).count()
        total = joined_data.count()

        self.logger.info(
            "Joined %s / %s recipes (%.1f%%)\n",
            f"{matched:,}",
            f"{total:,}",
            100.0 * matched / total if total > 0 else 0,
        )

        return joined_data

    def write_output(self, final_data) -> int:
        self.logger.info("STEP 5/5: Writing Output")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        output_data = final_data.map(remove_internal_fields)
        count = output_data.count()

        output_df = output_data.map(lambda x: Row(**x)).toDF()

        temp_path = self.output_path + ".tmp"
        output_df.coalesce(1).write.mode("overwrite").json(temp_path)

        part_files = [
            f
            for f in os.listdir(temp_path)
            if f.startswith("part-") and f.endswith(".json")
        ]
        temp_path = self.output_path + ".tmp"
        output_df.coalesce(1).write.mode("overwrite").json(temp_path)

        part_files = [f for f in os.listdir(temp_path) if f.startswith('part-') and f.endswith('.json')]
        if part_files:
            shutil.move(os.path.join(temp_path, part_files[0]), self.output_path)
            shutil.rmtree(temp_path, ignore_errors=True)

        self.logger.info("Wrote %s records to %s\n", f"{count:,}", self.output_path)
        return count


def main() -> None:
    spark = create_spark_session()
    mapper = WikipediaRecipeMapper(
        spark,
        wiki_dump_path=config.WIKI_DUMP_PATH,
        wiki_index_path=config.WIKI_INDEX_PATH,
        output_path=config.WIKI_RECIPES_OUTPUT,
        recipes_path=config.RECIPES_FILE,
    )
    mapper.run()
    spark.stop()


if __name__ == "__main__":
    main()
