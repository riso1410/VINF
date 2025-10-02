import os
import json
import re
import time
import html
from dataclasses import dataclass, field
from typing import List, Optional
import config

@dataclass
class RecipeMetadata:
    """Recipe object structure"""
    url: str = ""
    html_file: str = ""
    title: str = ""
    description: str = ""
    ingredients: List[str] = field(default_factory=list)
    method: str = ""
    prep_time: str = ""
    servings: str = ""
    difficulty: str = ""
    chef: str = ""

class RecipeScraper:
    def __init__(self, html_dir: str = None, output_dir: str = None, urls_file: str = None):
        # Use config defaults if not specified
        self.html_dir = html_dir if html_dir is not None else config.RAW_HTML_DIR
        self.output_dir = output_dir if output_dir is not None else config.SCRAPED_DIR
        self.urls_file = urls_file if urls_file is not None else config.URLS_FILE
        self.recipes_file = os.path.join(self.output_dir, "recipes.jsonl")
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Setup logging
        self.logger = config.setup_logging(config.SCRAPER_LOG)

    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and normalizing whitespace"""
        if not text:
            return ""
        
        # Remove script and style elements completely
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_json_ld_recipe(self, html_content: str) -> dict:
        """Extract Recipe from JSON-LD structured data"""
        # First try standard JSON-LD script tags
        pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                # Clean the match
                clean_match = match.strip()
                data = json.loads(clean_match)
                
                # Handle array format
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Recipe':
                            return item
                elif isinstance(data, dict) and data.get('@type') == 'Recipe':
                    return data
                    
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse JSON-LD: {e}")
                continue
        
        # Try extracting from embedded JavaScript objects and window.__INITIAL_STATE__
        return self.extract_embedded_recipe_data(html_content)

    def extract_embedded_recipe_data(self, html_content: str) -> dict:
        """Extract recipe data from embedded JavaScript objects"""
        # Look for recipe data in JavaScript variables or objects
        patterns = [
            # Pattern for window.__INITIAL_STATE__ or similar with recipe data
            r'window\.__INITIAL_STATE__\s*=\s*({.*?"difficulty"\s*:\s*\d+.*?});',
            r'window\.__INITIAL_STATE__\s*=\s*({.*?"@type"\s*:\s*"Recipe".*?});',
            # Pattern for "schema":[{...,"@type":"Recipe",...}]
            r'"schema"\s*:\s*\[\s*{[^}]*"@context"\s*:\s*"[^"]*schema\.org"[^}]*"@type"\s*:\s*"Recipe"[^}]*}[^}]*}[^}]*}[^}]*}[^}]*}[^}]*}[^}]*}[^}]*}\s*\]',
            # Pattern for data objects containing recipe
            r'["\']recipe["\']\s*:\s*({.*?"@type"\s*:\s*"Recipe".*?})',
            # Broader pattern for recipe data in JS objects
            r'"recipe"\s*:\s*({.*?"difficulty"\s*:\s*(?:\d+|null).*?})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    # Try to extract the recipe object from the match
                    recipe_data = self.extract_recipe_from_js_match(match)
                    if recipe_data:
                        return recipe_data
                except Exception as e:
                    self.logger.debug(f"Failed to parse embedded recipe data: {e}")
                    continue
        
        return {}

    def extract_recipe_from_js_match(self, js_text: str) -> dict:
        """Extract recipe data from JavaScript text match"""
        try:
            # Try to parse the entire match as JSON first
            try:
                data = json.loads(js_text)
                
                # If it's a complete object, look for recipe data within it
                if isinstance(data, dict):
                    # Check if it's directly a recipe
                    if data.get('@type') == 'Recipe':
                        return data
                    
                    # Look for recipe in nested structures
                    recipe = self.find_recipe_in_nested_object(data)
                    if recipe:
                        return recipe
                        
            except json.JSONDecodeError:
                pass
            
            # Look for the Recipe object within the JavaScript
            recipe_patterns = [
                r'"@type"\s*:\s*"Recipe"[^}]*(?:{[^}]*}[^}]*)*',
                r'"recipe"\s*:\s*({[^}]*"difficulty"[^}]*})',
                r'"difficulty"\s*:\s*\d+[^}]*'
            ]
            
            for pattern in recipe_patterns:
                potential_recipes = re.findall(pattern, js_text, re.IGNORECASE | re.DOTALL)
                
                for recipe_text in potential_recipes:
                    try:
                        start_pos = js_text.find(recipe_text)
                        if start_pos == -1:
                            continue
                        
                        # Find the complete object by counting braces
                        brace_count = 0
                        start_brace = js_text.rfind('{', 0, start_pos)
                        if start_brace == -1:
                            continue
                        
                        for i, char in enumerate(js_text[start_brace:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete object
                                    recipe_json = js_text[start_brace:start_brace + i + 1]
                                    
                                    # Try to parse as JSON
                                    try:
                                        recipe_data = json.loads(recipe_json)
                                        if (recipe_data.get('@type') == 'Recipe' or 
                                            'difficulty' in recipe_data or 
                                            'ingredients' in recipe_data):
                                            return recipe_data
                                    except json.JSONDecodeError:
                                        # Try to fix common JSON issues
                                        fixed_json = self.fix_js_object_to_json(recipe_json)
                                        if fixed_json:
                                            try:
                                                recipe_data = json.loads(fixed_json)
                                                if (recipe_data.get('@type') == 'Recipe' or 
                                                    'difficulty' in recipe_data or 
                                                    'ingredients' in recipe_data):
                                                    return recipe_data
                                            except json.JSONDecodeError:
                                                continue
                                    break
                                    
                    except Exception as e:
                        self.logger.debug(f"Error extracting recipe from JS: {e}")
                        continue
            
        except Exception as e:
            self.logger.debug(f"Error in extract_recipe_from_js_match: {e}")
        
        return {}
    
    def find_recipe_in_nested_object(self, obj: dict) -> dict:
        """Recursively search for recipe data in nested objects"""
        if isinstance(obj, dict):
            # Check if current object is a recipe
            if obj.get('@type') == 'Recipe':
                return obj
            
            # Check if it has recipe-like properties
            if ('difficulty' in obj and 'ingredients' in obj) or ('method_legacy' in obj):
                return obj
            
            # Search in nested objects
            for key, value in obj.items():
                if key == 'recipe' and isinstance(value, dict):
                    return value
                elif isinstance(value, dict):
                    result = self.find_recipe_in_nested_object(value)
                    if result:
                        return result
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result = self.find_recipe_in_nested_object(item)
                            if result:
                                return result
        
        return {}

    def fix_js_object_to_json(self, js_object: str) -> str:
        """Try to convert JavaScript object notation to valid JSON"""
        try:
            # Remove any trailing commas
            js_object = re.sub(r',(\s*[}\]])', r'\1', js_object)
            
            # Fix unquoted keys (basic attempt)
            js_object = re.sub(r'([{,]\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', js_object)
            
            # Fix single quotes to double quotes
            js_object = js_object.replace("'", '"')
            
            return js_object
        except Exception:
            return ""

    def extract_title(self, html_content: str, json_data: dict) -> str:
        """Extract recipe title"""
        # First try JSON-LD
        if json_data.get('name'):
            return self.clean_text(json_data['name'])
        
        # Fallback to HTML patterns
        patterns = [
            r'<h1[^>]*class="[^"]*recipe[^"]*title[^"]*"[^>]*>(.*?)</h1>',
            r'<h1[^>]*>(.*?)</h1>',
            r'<title[^>]*>(.*?)</title>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                return self.clean_text(match.group(1))
        
        return ""

    def extract_description(self, html_content: str, json_data: dict) -> str:
        """Extract recipe description"""
        # First try JSON-LD
        if json_data.get('description'):
            return self.clean_text(json_data['description'])
        
        # Fallback to meta description
        pattern = r'<meta[^>]*name="description"[^>]*content="([^"]*)"[^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return self.clean_text(match.group(1))
        
        return ""

    def extract_ingredients(self, html_content: str, json_data: dict) -> List[str]:
        """Extract ingredients from JSON-LD structured data"""
        ingredients = []
        
        # First try from passed json_data (if already extracted)
        if json_data and 'recipeIngredient' in json_data:
            recipe_ingredients = json_data['recipeIngredient']
            for ingredient in recipe_ingredients:
                ingredient_text = str(ingredient).strip()
                
                # Skip section headers (usually short and end with colon)
                if (ingredient_text and 
                    not ingredient_text.endswith(':') and 
                    len(ingredient_text) > 5 and
                    not ingredient_text.startswith('For the')):
                    ingredients.append(ingredient_text)
            
            if ingredients:
                return ingredients[:20]  # Limit to first 20
        
        # If not found in json_data, extract from HTML JSON-LD scripts
        try:
            json_ld_pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
            json_matches = re.findall(json_ld_pattern, html_content, re.DOTALL | re.IGNORECASE)
            
            for json_content in json_matches:
                try:
                    # Parse JSON
                    data = json.loads(json_content)
                    
                    # Check if this is recipe data
                    if isinstance(data, dict) and data.get('@type') == 'Recipe':
                        recipe_ingredients = data.get('recipeIngredient', [])
                        
                        for ingredient in recipe_ingredients:
                            ingredient_text = str(ingredient).strip()
                            
                            # Skip section headers (usually short and end with colon)
                            if (ingredient_text and 
                                not ingredient_text.endswith(':') and 
                                len(ingredient_text) > 5 and
                                not ingredient_text.startswith('For the')):
                                ingredients.append(ingredient_text)
                        
                        # If we found ingredients, break out of the loop
                        if ingredients:
                            break
                            
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    continue
        
        except Exception as e:
            self.logger.error(f"Error extracting ingredients from JSON-LD: {e}")
        
        return ingredients 

    def extract_method(self, html_content: str, json_data: dict) -> str:
        """Extract recipe method as clean text"""
        # First try JSON-LD
        instructions = json_data.get('recipeInstructions', json_data.get('recipemethod', []))
        if instructions:
            instruction_texts = []
            
            if isinstance(instructions, str):
                # Handle HTML-encoded instructions
                cleaned_instructions = self.clean_text(instructions)
                return cleaned_instructions
            elif isinstance(instructions, list):
                for instruction in instructions:
                    if isinstance(instruction, dict):
                        text = instruction.get('text', '')
                        if text:
                            instruction_texts.append(self.clean_text(text))
                    elif isinstance(instruction, str):
                        instruction_texts.append(self.clean_text(instruction))
                
                if instruction_texts:
                    return ' '.join(instruction_texts)
        
        # Try embedded recipe data
        embedded_data = self.extract_embedded_recipe_data(html_content)
        if embedded_data and 'method_legacy' in embedded_data:
            method_text = embedded_data['method_legacy']
            if method_text:
                return self.clean_text(method_text)
        
        # Fallback to HTML patterns
        patterns = [
            r'<ol[^>]*class="[^"]*instruction[^"]*"[^>]*>(.*?)</ol>',
            r'<ol[^>]*class="[^"]*method[^"]*"[^>]*>(.*?)</ol>',
            r'<div[^>]*class="[^"]*method[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*class="[^"]*recipe-method[^"]*"[^>]*>(.*?)</div>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                # Extract steps from list items or paragraphs
                step_matches = re.findall(r'<(?:li|p)[^>]*>(.*?)</(?:li|p)>', match.group(1), re.IGNORECASE | re.DOTALL)
                if step_matches:
                    steps = [self.clean_text(step) for step in step_matches if step.strip()]
                    return ' '.join(steps)
                else:
                    # If no specific steps found, return the cleaned content
                    return self.clean_text(match.group(1))
        
        return ""

    def extract_author(self, html_content: str, json_data: dict) -> str:
        """Extract recipe author/chef"""
        # First try JSON-LD
        author = json_data.get('author', {})
        if isinstance(author, dict):
            return self.clean_text(author.get('name', ''))
        elif isinstance(author, str):
            return self.clean_text(author)
        
        return ""

    def format_duration(self, duration_str: str) -> str:
        """Convert ISO 8601 duration format to readable format"""
        if not duration_str:
            return ""
        
        # Handle ISO 8601 format like PT15M, PT1H30M, PT45M
        if duration_str.startswith('PT'):
            duration = duration_str[2:]  # Remove 'PT' prefix
            
            # Extract hours and minutes
            hours = 0
            minutes = 0
            
            if 'H' in duration:
                hours_match = re.search(r'(\d+)H', duration)
                if hours_match:
                    hours = int(hours_match.group(1))
            
            if 'M' in duration:
                minutes_match = re.search(r'(\d+)M', duration)
                if minutes_match:
                    minutes = int(minutes_match.group(1))
            
            # Format the output
            result_parts = []
            if hours > 0:
                result_parts.append(f"{hours} hr" if hours == 1 else f"{hours} hrs")
            if minutes > 0:
                result_parts.append(f"{minutes} min")
            
            return " ".join(result_parts) if result_parts else duration_str
        
        return duration_str

    def extract_prep_time(self, html_content: str, json_data: dict) -> str:
        """Extract preparation time"""
        # Try JSON-LD fields
        for time_field in ['prepTime', 'totalTime', 'cookTime']:
            if json_data.get(time_field):
                raw_time = self.clean_text(json_data[time_field])
                return self.format_duration(raw_time)
        
        # Try embedded recipe data
        embedded_data = self.extract_embedded_recipe_data(html_content)
        if embedded_data:
            for time_field in ['total_time', 'total_time_formatted', 'prep_time']:
                if time_field in embedded_data and embedded_data[time_field]:
                    time_value = embedded_data[time_field]
                    if isinstance(time_value, (int, float)):
                        return f"{time_value} min"
                    elif isinstance(time_value, str):
                        return self.clean_text(time_value)
        
        # Try HTML patterns
        patterns = [
            r'<span[^>]*class="[^"]*(?:prep|total|cook)[-_]?time[^"]*"[^>]*>([^<]+)</span>',
            r'<div[^>]*class="[^"]*(?:prep|total|cook)[-_]?time[^"]*"[^>]*>([^<]+)</div>',
            r'"(?:prep|total|cook)Time"\s*:\s*"([^"]+)"',
            r'(?:Prep|Total|Cook)\s*[Tt]ime\s*:?\s*([^<\n]+)',
            r'PT(\d+)M',  # ISO 8601 format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                time_str = self.clean_text(match.group(1))
                return self.format_duration(time_str)
        
        return ""

    def extract_servings(self, html_content: str, json_data: dict) -> str:
        """Extract servings information"""
        # Try JSON-LD
        yield_value = json_data.get('recipeYield')
        if yield_value:
            if isinstance(yield_value, list):
                return self.clean_text(str(yield_value[0]))
            return self.clean_text(str(yield_value))
        
        # Try embedded recipe data
        embedded_data = self.extract_embedded_recipe_data(html_content)
        if embedded_data and 'servings' in embedded_data:
            servings = embedded_data['servings']
            if servings:
                return self.clean_text(str(servings))
        
        # Try HTML patterns
        patterns = [
            r'<span[^>]*class="[^"]*servings?[^"]*"[^>]*>([^<]+)</span>',
            r'<div[^>]*class="[^"]*servings?[^"]*"[^>]*>([^<]+)</div>',
            r'"servings?"\s*:\s*"?([^",}]+)"?',
            r'Serves?\s*:?\s*(\d+)',
            r'Makes?\s*:?\s*(\d+)',
            r'Portions?\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                return self.clean_text(match.group(1))
        
        return ""

    def extract_difficulty(self, html_content: str, json_data: dict) -> str:
        """Extract recipe difficulty level"""
        # Try to get difficulty from embedded JSON data first (most reliable for Food Network)
        embedded_data = self.extract_embedded_recipe_data(html_content)
        if embedded_data and 'difficulty' in embedded_data:
            difficulty = embedded_data['difficulty']
            return self.convert_numeric_difficulty_to_text(difficulty)
        
        # Try JSON-LD data
        if json_data and 'difficulty' in json_data:
            difficulty = json_data['difficulty']
            return self.convert_numeric_difficulty_to_text(difficulty)
        
        # Try specific difficulty pattern from HTML with icon
        pattern = r'<li[^>]*class="[^"]*p-2[^"]*"[^>]*>\s*<span[^>]*>\s*<img[^>]*difficulty-icon[^>]*>\s*<span[^>]*>\s*([^<]+)\s*</span>'
        match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        
        # Fallback HTML patterns for text-based difficulty
        patterns = [
            r'<span[^>]*class="[^"]*difficulty[^"]*"[^>]*>([^<]+)</span>',
            r'"difficulty"\s*:\s*"([^"]+)"',
            r'difficulty[^>]*>([^<]+)',
            r'<span[^>]*>\s*(Very Easy|Easy|Medium|Hard|Beginner|Intermediate|Advanced)\s*</span>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                difficulty_text = self.clean_text(match.group(1))
                # Check if it's a numeric string and convert it
                if difficulty_text.isdigit():
                    return self.convert_numeric_difficulty_to_text(int(difficulty_text))
                return difficulty_text
        
        return ""
    
    def convert_numeric_difficulty_to_text(self, difficulty) -> str:
        """Convert numeric difficulty to text labels"""
        if difficulty is None:
            return ""
        
        try:
            difficulty_num = int(difficulty)
            return config.DIFFICULTY_MAP.get(difficulty_num, str(difficulty))
        except (ValueError, TypeError):
            return str(difficulty) if difficulty else ""

    def get_html_file_path(self, url: str) -> Optional[str]:
        """Get the HTML file path for a given URL"""
        # Convert URL to filename - replace slashes with underscores to match crawler output
        filename = url.replace('https://', '').replace('http://', '')
        filename = filename.replace('/', '_')
        
        if not filename.endswith('.html'):
            filename += '.html'
        
        html_path = os.path.join(self.html_dir, filename)
        
        if os.path.exists(html_path):
            return html_path
        
        return None

    def extract_recipe_metadata(self, url: str) -> Optional[RecipeMetadata]:
        """Extract clean recipe metadata from HTML content"""
        html_file = self.get_html_file_path(url)
        if not html_file:
            self.logger.warning(f"HTML file not found for URL: {url}")
            return None
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read HTML file {html_file}: {e}")
            return None
        
        # Extract JSON-LD data first
        json_data = self.extract_json_ld_recipe(html_content)
        
        # Create metadata object with extracted data
        # Convert Windows path separators to forward slashes for consistent output
        normalized_html_file = html_file.replace('\\', '/')
        
        metadata = RecipeMetadata(
            url=url,
            html_file=normalized_html_file,
            title=self.extract_title(html_content, json_data),
            description=self.extract_description(html_content, json_data),
            method=self.extract_method(html_content, json_data),
            ingredients=self.extract_ingredients(html_content, json_data),
            prep_time=self.extract_prep_time(html_content, json_data),
            servings=self.extract_servings(html_content, json_data),
            difficulty=self.extract_difficulty(html_content, json_data),
            chef=self.extract_author(html_content, json_data)
        )
        
        return metadata

    def save_to_jsonl(self, metadata: RecipeMetadata):
        """Save single recipe to the main recipes.jsonl file"""
        try:
            # Convert to dict
            data_dict = {
                'url': metadata.url,
                'html_file': metadata.html_file,
                'title': metadata.title,
                'description': metadata.description,
                'method': metadata.method,
                'ingredients': metadata.ingredients,
                'prep_time': metadata.prep_time,
                'servings': metadata.servings,
                'difficulty': metadata.difficulty,
                'chef': metadata.chef
            }
            
            # Append to the single JSONL file
            with open(self.recipes_file, 'a', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False)
                f.write('\n')
            
        except Exception as e:
            self.logger.error(f"Error saving recipe to {self.recipes_file}: {e}")

    def scrape_all_recipes(self):
        """Main method to scrape all recipes to a single recipes.jsonl file"""
        if not os.path.exists(self.urls_file):
            self.logger.error(f"URLs file not found: {self.urls_file}")
            return
        
        try:
            with open(self.urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Error reading URLs file: {e}")
            return
        
        # Clear the recipes file if it exists to start fresh
        if os.path.exists(self.recipes_file):
            os.remove(self.recipes_file)
        
        self.logger.info(f"Starting to scrape {len(urls)} URLs")
        self.logger.info(f"Saving all recipes to: {self.recipes_file}")
        
        processed_count = 0
        successful_count = 0
        
        for url in urls:
            try:
                self.logger.info(f"Processing URL {processed_count + 1}/{len(urls)}: {url}")
                
                metadata = self.extract_recipe_metadata(url)
                if metadata:
                    # Save to single JSONL file
                    self.save_to_jsonl(metadata)
                    successful_count += 1
                    
                    if successful_count % config.SCRAPER_PROGRESS_INTERVAL == 0:
                        self.logger.info(f"Successfully processed {successful_count} recipes so far...")
                
                processed_count += 1
                
                # Rate limiting
                time.sleep(config.SCRAPER_RATE_LIMIT)
                
            except Exception as e:
                self.logger.error(f"Error processing URL {url}: {e}")
                continue
        
        self.logger.info(f"Scraping completed. Processed {processed_count} URLs, successfully extracted {successful_count} recipes")
        self.logger.info(f"All recipes saved to: {self.recipes_file}")

def main():
    scraper = RecipeScraper()
    scraper.scrape_all_recipes()

if __name__ == "__main__":
    main()