import bz2
import json
import re
from pathlib import Path

import config

WIKIPEDIA_PATTERNS = {
    "abstract": re.compile(
        r"<text[^>]*>(?:(?!{{Infobox|{{Short description)[\s\S]*?)"
        r"([^{|\n][^\n]*?(?:\n(?![=\n]).*?)*?)"
        r"(?=\n\n|==|$)",
        re.DOTALL | re.MULTILINE,
    ),
    "wiki_links": re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"),
    "refs": re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL),
    "html_tags": re.compile(r"<[^>]+>"),
    "bold_italic": re.compile(r"'''?([^']+)'''?"),
    "templates": re.compile(r"{{[^}]*}}"),
    "multiple_spaces": re.compile(r"\s+"),
    "infobox": re.compile(r"{{Infobox[^}]*", re.IGNORECASE | re.DOTALL),
    "category": re.compile(r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", re.IGNORECASE),
}


def extract_wiki_data(xml_text: str) -> dict:
    data = {
        "wiki_description": None,
        "wiki_course": None,
        "wiki_origin": None,
        "wiki_serving_temp": None,
        "wiki_categories": [],
    }

    match = WIKIPEDIA_PATTERNS["abstract"].search(xml_text)
    if match:
        description = match.group(1)
        description = WIKIPEDIA_PATTERNS["refs"].sub("", description)
        description = WIKIPEDIA_PATTERNS["templates"].sub("", description)
        description = WIKIPEDIA_PATTERNS["wiki_links"].sub(r"\1", description)
        description = WIKIPEDIA_PATTERNS["bold_italic"].sub(r"\1", description)
        description = WIKIPEDIA_PATTERNS["html_tags"].sub("", description)
        description = WIKIPEDIA_PATTERNS["multiple_spaces"].sub(" ", description)
        data["wiki_description"] = description.strip()

    infobox_match = WIKIPEDIA_PATTERNS["infobox"].search(xml_text)
    if infobox_match:
        infobox_text = infobox_match.group(0)

        def extract_field(field_names):
            for name in field_names:
                pattern = re.compile(
                    r"\|\s*" + name + r"\s*=\s*([^|\n]*)", re.IGNORECASE
                )
                match = pattern.search(infobox_text)
                if match:
                    val = match.group(1).strip()
                    val = WIKIPEDIA_PATTERNS["wiki_links"].sub(r"\1", val)
                    val = WIKIPEDIA_PATTERNS["html_tags"].sub("", val)
                    return val
            return None

        data["wiki_course"] = extract_field(["course"])
        data["wiki_origin"] = extract_field(["place_of_origin", "origin", "country"])
        data["wiki_serving_temp"] = extract_field(
            ["serving_temperature", "temperature"]
        )

    categories = WIKIPEDIA_PATTERNS["category"].findall(xml_text)
    if categories:
        data["wiki_categories"] = [c.strip() for c in categories]

    return data


def read_wikipedia_article_at_offset(dump_path: Path, offset: int) -> str | None:
    try:
        with bz2.open(dump_path, "rt", encoding="utf-8") as f:
            current_pos = 0

            for line in f:
                if current_pos >= offset:
                    article_lines = [line]

                    for line in f:
                        article_lines.append(line)
                        if "</page>" in line:
                            break

                    return "".join(article_lines)

                current_pos += len(line.encode("utf-8"))

        return None

    except Exception as e:
        print(f"Error reading offset {offset}: {e}")
        return None


def get_wiki_data(dump_path: Path, offset: int) -> dict | None:
    article_xml = read_wikipedia_article_at_offset(dump_path, offset)

    if not article_xml:
        return None

    return extract_wiki_data(article_xml)


def extract_descriptions_batch(
    matches_file: Path, dump_path: Path, output_file: Path, limit: int = None
):
    print(f"Loading matches from: {matches_file}")
    matches = []

    with open(matches_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            matches.append(json.loads(line))

    print(f"Processing {len(matches)} matches...")

    enriched_matches = []

    for i, match in enumerate(matches):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(matches)} matches...")

        offset = match.get("wiki_offset")
        if offset:
            wiki_data = get_wiki_data(dump_path, offset)
            if wiki_data:
                match.update(wiki_data)

        enriched_matches.append(match)

    print(f"\nSaving enriched matches to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for match in enriched_matches:
            f.write(json.dumps(match) + "\n")

    print(f"Saved {len(enriched_matches)} enriched matches")


def main():
    matches_file = Path(config.INDEX_DIR) / "wiki_matches.jsonl"
    dump_path = Path(config.WIKI_DUMP_PATH)
    output_file = Path(config.INDEX_DIR) / "wiki_matches_enriched.jsonl"

    if not matches_file.exists():
        print(f"Error: {matches_file} not found!")
        return

    if not dump_path.exists():
        print(f"Error: Wikipedia dump not found at {dump_path}")
        print("Please download the Wikipedia dump first.")
        return

    print("=" * 70)
    print("WIKIPEDIA DESCRIPTION EXTRACTION")
    print("=" * 70)
    print("\nTesting on first 10 matches...")
    print("(Change limit=None to process all matches)\n")

    extract_descriptions_batch(
        matches_file=matches_file,
        dump_path=dump_path,
        output_file=output_file,
        limit=10,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
