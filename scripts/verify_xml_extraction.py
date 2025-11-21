import re
import glob
import os
from pathlib import Path


def extract_wiki_data_from_xml(article_xml):
    """
    Extract description, infobox fields, and categories from article XML.
    Returns a dictionary of extracted data.
    """
    # Compile regex patterns
    lead_section_pattern = re.compile(r"^(.*?)(?=\n={2,})", re.DOTALL)
    redirect_pattern = re.compile(r"#REDIRECT\s*\[\[([^\]]+)\]\]", re.IGNORECASE)

    # Cleaning patterns
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

    # 0. Check for Redirect
    if redirect_pattern.search(article_xml):
        data["is_redirect"] = True
        return data

    # 1. Extract Description
    # Priority 1: Description section
    # Look for == Description == (case insensitive) and take content until next section or end
    description_section_pattern = re.compile(
        r"={2,}\s*Description\s*={2,}(.*?)(?=\n={2,}|$)", re.IGNORECASE | re.DOTALL
    )
    match_desc = description_section_pattern.search(article_xml)

    description = None
    if match_desc:
        description = match_desc.group(1)

    # Priority 2: Lead Section (fallback)
    if not description:
        match = lead_section_pattern.search(article_xml)
        if match:
            description = match.group(1)
        else:
            description = article_xml[:2000]

    # Clean description
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

    # 2. Extract Infobox Fields
    infobox_match = re.search(r"{{Infobox\s+([^\n]+)", article_xml, re.IGNORECASE)
    if infobox_match:
        infobox_context = article_xml[:5000]

        def extract_field(field_names):
            for name in field_names:
                # Improved regex: look for value until next line starting with | or }}
                # This handles [[Link|Text]] correctly by not stopping at the internal pipe
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

    # 3. Extract Categories
    categories = re.findall(
        r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", article_xml, re.IGNORECASE
    )
    if categories:
        data["wiki_categories"] = [c.strip() for c in categories]

    return data


def test_pampushka_extraction():
    print("\n--- Testing Pampushka Example ---")
    pampushka_xml = """
{{Infobox food
| name = Pampushka
| image = Pampushky-plain.jpg
| image_size = 250px
| caption = Plain ''pampushky''
| alt = Seven pampushky on a plate
| alternate_name = 
| country = 

[[Ukraine]]

| region = 
| national_cuisine = 
| creator =           <!-- or | creators = -->
| year = 
| mintime = 
| maxtime = 
| type = 
| course = 
| served = 
| main_ingredient = 
| minor_ingredient = 
| variations = 
| serving_size =
| calories = 
| protein = 
| fat = 
| carbohydrate = 
| glycemic_index = 
| similar_dish = 
| other = 
}}
    """
    data = extract_wiki_data_from_xml(pampushka_xml)
    print(f"Country extracted: '{data['wiki_origin']}'")
    expected = "Ukraine"
    if data["wiki_origin"] == expected:
        print("SUCCESS: Pampushka country extracted correctly.")
    else:
        print(f"FAILURE: Expected '{expected}', got '{data['wiki_origin']}'")


def test_description_section():
    print("\n--- Testing Description Section Priority ---")
    xml = """
'''Pampushka''' is a food.

==Description==
Pampushka is a small savory bun.
It is delicious.

==History==
Old food.
    """
    data = extract_wiki_data_from_xml(xml)
    print(f"Description extracted: '{data['wiki_description']}'")
    expected_start = "Pampushka is a small savory bun"
    if data["wiki_description"] and data["wiki_description"].startswith(expected_start):
        print("SUCCESS: Description section extracted.")
    else:
        print(
            f"FAILURE: Expected start '{expected_start}', got '{data['wiki_description']}'"
        )


def main():
    # 1. Test Hardcoded Examples
    test_pampushka_extraction()
    test_description_section()

    # 2. Test Real Files
    # Check 47290.xml specifically as it's the user's active file
    files_to_check = ["data/xml/47290.xml"]

    # Handle relative paths if running from scripts dir
    if not os.path.exists("data"):
        if os.path.exists("../data"):
            files_to_check = ["../" + f for f in files_to_check]

    for xml_file in files_to_check:
        print(f"\n--- Checking file: {xml_file} ---", flush=True)

        if os.path.exists(xml_file):
            try:
                with open(xml_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title roughly for context
                title_match = re.search(r"<title>(.*?)</title>", content)
                title = title_match.group(1) if title_match else "Unknown"
                print(f"Title: {title}", flush=True)

                data = extract_wiki_data_from_xml(content)

                print(f"Redirect: {data['is_redirect']}", flush=True)
                if not data["is_redirect"]:
                    print(
                        f"Description: {data['wiki_description'][:500]}..."
                        if data["wiki_description"]
                        else "Description: None",
                        flush=True,
                    )
                    print(f"Course: {data['wiki_course']}", flush=True)
                    print(f"Origin: {data['wiki_origin']}", flush=True)
                    print(f"Temp: {data['wiki_serving_temp']}", flush=True)
                    print(f"Categories: {data['wiki_categories'][:5]}", flush=True)

            except Exception as e:
                print(f"Error processing {xml_file}: {e}", flush=True)
        else:
            print(f"File not found: {xml_file}", flush=True)


if __name__ == "__main__":
    main()
