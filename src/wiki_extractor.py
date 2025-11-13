"""
Wikipedia Description Extractor
Extracts the first paragraph/abstract from Wikipedia articles using the dump file.
"""

import re
import bz2
import json
from pathlib import Path
import config


# Regex patterns for extracting Wikipedia content
WIKIPEDIA_PATTERNS = {
    # Extract first paragraph (abstract) - text between <text> and first \n\n
    'abstract': re.compile(
        r'<text[^>]*>(?:(?!{{Infobox|{{Short description)[\s\S]*?)'  # Skip infoboxes
        r'([^{|\n][^\n]*?(?:\n(?![=\n]).*?)*?)'  # First paragraph
        r'(?=\n\n|==|$)',
        re.DOTALL | re.MULTILINE
    ),
    
    # Clean wiki markup
    'wiki_links': re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]'),  # [[link|text]] -> text
    'refs': re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL),  # Remove references
    'html_tags': re.compile(r'<[^>]+>'),  # Remove HTML tags
    'bold_italic': re.compile(r"'''?([^']+)'''?"),  # '''bold''' -> bold
    'templates': re.compile(r'{{[^}]*}}'),  # Remove templates
    'multiple_spaces': re.compile(r'\s+'),  # Collapse spaces
}


def extract_description_from_xml(xml_text: str) -> str | None:
    """
    Extract clean description from Wikipedia XML content.
    
    Args:
        xml_text: Raw Wikipedia article XML
        
    Returns:
        Clean description text or None
    """
    # Extract first paragraph
    match = WIKIPEDIA_PATTERNS['abstract'].search(xml_text)
    if not match:
        return None
    
    description = match.group(1)
    
    # Clean wiki markup
    description = WIKIPEDIA_PATTERNS['refs'].sub('', description)  # Remove refs
    description = WIKIPEDIA_PATTERNS['templates'].sub('', description)  # Remove templates
    description = WIKIPEDIA_PATTERNS['wiki_links'].sub(r'\1', description)  # Clean links
    description = WIKIPEDIA_PATTERNS['bold_italic'].sub(r'\1', description)  # Remove bold/italic
    description = WIKIPEDIA_PATTERNS['html_tags'].sub('', description)  # Remove HTML
    description = WIKIPEDIA_PATTERNS['multiple_spaces'].sub(' ', description)  # Clean spaces
    
    # Clean up
    description = description.strip()
    
    # Only return if we have substantial content (at least 50 chars)
    if len(description) < 50:
        return None
    
    return description


def read_wikipedia_article_at_offset(dump_path: Path, offset: int) -> str | None:
    """
    Read Wikipedia article at specific byte offset in dump file.
    
    Args:
        dump_path: Path to Wikipedia dump file (bz2)
        offset: Byte offset in the file
        
    Returns:
        Article XML text or None
    """
    try:
        with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
            # Seek to offset (note: seeking in bz2 is slow, reads from start)
            # For production, consider using multistream index blocks
            current_pos = 0
            
            # Read until we find the article
            for line in f:
                if current_pos >= offset:
                    # We've reached the offset, now read the article
                    article_lines = [line]
                    
                    # Read until end of article (</page>)
                    for line in f:
                        article_lines.append(line)
                        if '</page>' in line:
                            break
                    
                    return ''.join(article_lines)
                
                current_pos += len(line.encode('utf-8'))
        
        return None
        
    except Exception as e:
        print(f"Error reading offset {offset}: {e}")
        return None


def get_wiki_description(dump_path: Path, offset: int) -> str | None:
    """
    Get Wikipedia description at given offset in dump file.
    
    This is a convenience function that combines reading and extracting.
    
    Args:
        dump_path: Path to Wikipedia dump file (bz2)
        offset: Byte offset in the file
        
    Returns:
        Clean description text or None if extraction fails
    """
    article_xml = read_wikipedia_article_at_offset(dump_path, offset)
    
    if not article_xml:
        return None
    
    return extract_description_from_xml(article_xml)


def extract_descriptions_batch(matches_file: Path, dump_path: Path, output_file: Path, limit: int = None):
    """
    Extract Wikipedia descriptions for all matches.
    
    Args:
        matches_file: Path to wiki_matches.jsonl
        dump_path: Path to Wikipedia dump
        output_file: Path to save enriched matches
        limit: Optional limit for testing (number of matches to process)
    """
    print(f"Loading matches from: {matches_file}")
    matches = []
    
    with open(matches_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            matches.append(json.loads(line))
    
    print(f"Processing {len(matches)} matches...")
    
    enriched_matches = []
    
    for i, match in enumerate(matches):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(matches)} matches...")
        
        # Get Wikipedia description using the helper function
        offset = match.get('wiki_offset')
        if offset:
            description = get_wiki_description(dump_path, offset)
            match['wiki_description'] = description
        
        enriched_matches.append(match)
    
    # Save enriched matches
    print(f"\nSaving enriched matches to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for match in enriched_matches:
            f.write(json.dumps(match) + '\n')
    
    print(f"Saved {len(enriched_matches)} enriched matches")


def main():
    """Test the extraction on a few examples."""
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
    
    # Extract descriptions (test on first 10 for now)
    print("=" * 70)
    print("WIKIPEDIA DESCRIPTION EXTRACTION")
    print("=" * 70)
    print("\nTesting on first 10 matches...")
    print("(Change limit=None to process all matches)\n")
    
    extract_descriptions_batch(
        matches_file=matches_file,
        dump_path=dump_path,
        output_file=output_file,
        limit=10  # Change to None to process all
    )
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
