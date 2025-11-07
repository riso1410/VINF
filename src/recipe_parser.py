import os
import re
from dataclasses import dataclass, field
from typing import Optional

from markitdown import MarkItDown


@dataclass
class RecipeMetadata:
    url: str = ""
    html_file: str = ""
    title: str = ""
    description: str = ""
    ingredients: list[str] = field(default_factory=list)
    method: str = ""
    prep_time: str = ""
    servings: str = ""
    difficulty: str = ""
    chef: str = ""


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_filename_to_url(filename: str) -> str:
    url = filename.replace(".html", "")
    url = url.replace("_", "/")
    if not url.startswith("http"):
        url = "https://" + url
    return url


def url_to_html_filename(url: str) -> str:
    filename = url.replace("https://", "").replace("http://", "")
    filename = filename.replace("/", "_")
    if not filename.endswith(".html"):
        filename += ".html"
    return filename


def url_to_html_path(url: str, base_dir: str) -> Optional[str]:
    filename = url_to_html_filename(url)
    html_path = os.path.join(base_dir, filename)
    if os.path.exists(html_path):
        return html_path
    return None


def extract_title(markdown_content: str) -> str:
    match = re.search(r"^# (.+)$", markdown_content, re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    return ""


def extract_description(markdown_content: str) -> str:
    patterns = [
        r"\[Rate\]\(#vote\).*?(?:\n\s*\* !\[A fallback image for Food Network UK\][^\n]*)+\s*\n\s*(.*?)(?=\n{2,}|^## |!\[A fallback image for Food Network UK\]|Featured In:)",
        r"\[Rate\]\(#vote\)\s*\n\s*!\[[^\]]*\]\([^\)]*\)\s*\n\s*(.*?)(?=\n{2,}|^## |!\[A fallback image for Food Network UK\]|Featured In:)",
        r"\[Rate\]\(#vote\)\s*\n\s*(.*?)(?=\n{2,}|^## |!\[A fallback image for Food Network UK\]|Featured In:)",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown_content, re.DOTALL | re.MULTILINE)
        if match:
            description = match.group(1).strip()
            description = re.sub(r"!\[.*?\]\(.*?\)", "", description)
            description = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", description)
            return clean_text(description)
    return ""


def extract_ingredients(markdown_content: str) -> list[str]:
    ingredients: list[str] = []
    match = re.search(
        r"## Ingredients\s*\n(.*?)(?=\n## |\n\nRead More|$)",
        markdown_content,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ingredients

    ingredients_section = match.group(1)
    matches = re.findall(r"\* \[ \] (.+?)(?=\n|$)", ingredients_section)
    for ingredient in matches:
        ingredient_text = clean_text(ingredient)
        if not ingredient_text.endswith(":"):
            ingredients.append(ingredient_text)
    return ingredients


def extract_method(markdown_content: str) -> str:
    match = re.search(
        r"## Method\s*\n(.*?)(?=\n\*Copyright|\n\*From Food Network|\nRead More\s*\n\s*Rate this recipe|## Related Recipes)",
        markdown_content,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    method_text = match.group(1).strip()
    method_text = re.sub(r"\s*Read More\s*$", "", method_text, flags=re.IGNORECASE)
    return clean_text(method_text)


def extract_author(markdown_content: str) -> str:
    match = re.search(
        r'\[!\[\]\(\)\s*\n([^)\[]+)\]\(https://foodnetwork\.co\.uk/chefs/[^\)]*"Go to Author"\)\s*\n(?:!\[A fallback image for Food Network UK\]\(/images/illustrations/meal-2\.svg\))?(?:!\[A fallback image for Food Network UK\]\(/images/illustrations/leaf\.svg\))?(?:!\[A fallback image for Food Network UK\]\(/images/illustrations/stirring\.svg\))?',
        markdown_content,
        re.DOTALL,
    )
    if match:
        return clean_text(match.group(1))
    return ""


def extract_prep_time(markdown_content: str) -> str:
    match = re.search(
        r"!\[A fallback image for Food Network UK\]\(/images/time-icon\.svg\)((?:\d+\s+HRS?)?\s*\d+\s+(?:MINS?|HRS?))",
        markdown_content,
        re.IGNORECASE,
    )
    if not match:
        return ""
    time_str = match.group(1).strip()
    time_str = re.sub(r"\bMINS?\b", "min", time_str, flags=re.IGNORECASE)
    time_str = re.sub(r"\bHRS?\b", "hr", time_str, flags=re.IGNORECASE)
    return clean_text(time_str)


def extract_servings(markdown_content: str) -> str:
    match = re.search(
        r"!\[A fallback image for Food Network UK\]\(/images/serves-icon\.svg\)(\d+)",
        markdown_content,
    )
    if match:
        return clean_text(match.group(1))
    return ""


def extract_difficulty(markdown_content: str) -> str:
    match = re.search(
        r"!\[A fallback image for Food Network UK\]\(/images/difficulty-icon\.svg\)([A-Za-z\s]+?)(?=\n|!\[)",
        markdown_content,
    )
    if not match:
        return ""
    difficulty = clean_text(match.group(1))
    if difficulty.isdigit() or len(difficulty) >= 50:
        return ""
    return difficulty


def normalise_html_path(path: str) -> str:
    return path.replace("\\", "/")


def should_skip_metadata(metadata: RecipeMetadata) -> bool:
    return metadata.title.strip() == "Recipes"


def metadata_to_dict(metadata: RecipeMetadata) -> dict:
    def fix_field(value, default):
        if value is None or value == {}:
            return default
        return value

    return {
        "url": fix_field(metadata.url, ""),
        "html_file": fix_field(metadata.html_file, ""),
        "title": fix_field(metadata.title, ""),
        "description": fix_field(metadata.description, ""),
        "method": fix_field(metadata.method, ""),
        "ingredients": fix_field(metadata.ingredients, []),
        "prep_time": fix_field(metadata.prep_time, ""),
        "servings": fix_field(metadata.servings, ""),
        "difficulty": fix_field(metadata.difficulty, ""),
        "chef": fix_field(metadata.chef, ""),
    }


def extract_recipe_from_markdown(
    markdown_content: str,
    *,
    url: str,
    html_file: str,
) -> RecipeMetadata:
    return RecipeMetadata(
        url=url,
        html_file=normalise_html_path(html_file),
        title=extract_title(markdown_content),
        description=extract_description(markdown_content),
        method=extract_method(markdown_content),
        ingredients=extract_ingredients(markdown_content),
        prep_time=extract_prep_time(markdown_content),
        servings=extract_servings(markdown_content),
        difficulty=extract_difficulty(markdown_content),
        chef=extract_author(markdown_content),
    )


def parse_recipe_html(
    html_file: str,
    *,
    url: Optional[str] = None,
    markdown_converter: Optional[MarkItDown] = None,
    logger=None,
) -> Optional[RecipeMetadata]:
    if not os.path.exists(html_file):
        if logger:
            logger.warning("HTML file not found: %s", html_file)
        return None

    resolved_url = url
    if not resolved_url:
        resolved_url = html_filename_to_url(os.path.basename(html_file))

    try:
        converter = markdown_converter or MarkItDown()
        markdown_result = converter.convert(html_file)
    except Exception as exc:
        if logger:
            logger.warning("Failed to convert %s to markdown: %s", html_file, exc)
        return None

    markdown_content = getattr(markdown_result, "text_content", "") or ""
    if not markdown_content:
        if logger:
            logger.warning("Markdown conversion returned empty content for %s", html_file)
        return None

    return extract_recipe_from_markdown(
        markdown_content,
        url=resolved_url,
        html_file=html_file,
    )
