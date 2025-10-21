import argparse
import os
import pickle as pkl
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from selenium import webdriver

import config


def html_filename_to_url(filename: str) -> str:
    url = filename.replace(".html", "")
    url = url.replace("_", "/")
    if not url.startswith("http"):
        url = "https://" + url
    return url


class Crawler:
    def __init__(self, start_url: str = config.START_URL):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.RAW_HTML_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)

        self.logger = config.setup_logging(config.CRAWLER_LOG)

        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.urls_to_visit = deque([start_url])
        self.visited_urls = set()
        self.downloaded_recipes = set()
        self.recipes_saved = 0

        self.driver = None

        self.setup_selenium()

        self.checkpoint_file = config.CHECKPOINT_FILE
        self.load_checkpoint()

    def setup_selenium(self):
        chrome_options = config.get_chrome_options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
        self.driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "rb") as f:
                    checkpoint = pkl.load(f)
                    self.urls_to_visit = checkpoint.get(
                        "urls_to_visit", [self.start_url]
                    )
                    self.visited_urls = checkpoint.get("visited_urls", set())
                    self.downloaded_recipes = checkpoint.get(
                        "downloaded_recipes", set()
                    )
                    self.recipes_saved = len(self.downloaded_recipes)
                self.logger.info(
                    f"Loaded checkpoint: {len(self.visited_urls)} pages visited, {self.recipes_saved} recipes downloaded"
                )
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")

    def save_checkpoint(self):
        checkpoint = {
            "urls_to_visit": list(self.urls_to_visit),
            "visited_urls": self.visited_urls,
            "downloaded_recipes": self.downloaded_recipes,
        }
        with open(self.checkpoint_file, "wb") as f:
            pkl.dump(checkpoint, f)

    def save_html_content(self, url, html_content):
        parsed = urlparse(url)

        url_filename = re.sub(r"[^\w\-_.]", "_", parsed.path.strip("/"))
        domain_prefix = re.sub(r"[^\w\-_.]", "_", parsed.netloc)

        filename = f"{domain_prefix}_{url_filename}.html"

        filepath = os.path.join(config.RAW_HTML_DIR, filename)

        if os.path.exists(filepath):
            self.logger.debug(f"File already exists, skipping: {filename}")
            return None

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    def is_recipe_url(self, url):
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(path_parts) != 2:
            return False

        if path_parts[0] != "recipes":
            return False

        return True

    def extract_urls_from_html(self, html_content, base_url):
        urls = set()
        matches = re.findall(config.URL_PATTERN, html_content, re.IGNORECASE)

        for match in matches:
            absolute_url = urljoin(base_url, match)
            clean_url = absolute_url.split("#")[0]
            parsed = urlparse(clean_url)

            if parsed.netloc != self.domain:
                continue
            if any(clean_url.lower().endswith(ext) for ext in config.SKIP_EXTENSIONS):
                continue

            if self.domain == "foodnetwork.co.uk" and "/search?" in parsed.path + (
                "?" + parsed.query if parsed.query else ""
            ):
                continue

            urls.add(clean_url)
        return urls

    def handle_retry_logic(self, url, retry_count, max_retries):
        if retry_count >= max_retries:
            self.logger.error(f"Max retries exceeded for {url}")
            return False

        self.logger.info(f"Retrying {url} (attempt {retry_count + 1}/{max_retries})")
        time.sleep(3)
        self.crawl_page(url, retry_count + 1, max_retries)
        return True

    def crawl_page(self, url, retry_count=0, max_retries=None):
        if max_retries is None:
            max_retries = config.MAX_RETRIES

        try:
            self.logger.info(f"Crawling: {url}")

            self.driver.get(url)

            time.sleep(3)

            html_content = self.driver.page_source

            if self.is_recipe_url(url):
                if url not in self.downloaded_recipes:
                    saved_path = self.save_html_content(url, html_content)
                    if saved_path:
                        self.downloaded_recipes.add(url)
                        self.recipes_saved += 1
                        self.logger.info(f"Saved recipe {self.recipes_saved}: {url}")

            urls = self.extract_urls_from_html(html_content, url)

            new_recipe_urls = 0
            new_urls = sum(
                1 for found_url in urls if found_url not in self.visited_urls
            )
            new_recipe_urls = sum(
                1
                for found_url in urls
                if found_url not in self.visited_urls and self.is_recipe_url(found_url)
            )

            for found_url in urls:
                if found_url not in self.visited_urls:
                    self.urls_to_visit.append(found_url)

            self.logger.info(
                f"Found {new_urls} new URLs ({new_recipe_urls} recipes) on {url}"
            )

        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            self.handle_retry_logic(url, retry_count, max_retries)

    def crawl(self):
        self.logger.info(f"Starting BFS crawl from {self.start_url}")
        crawled_count = 0

        while self.urls_to_visit:
            current_url = self.urls_to_visit.popleft()

            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)
            crawled_count += 1

            self.crawl_page(current_url)

            if crawled_count % 10 == 0:
                self.save_checkpoint()
                self.logger.info(
                    f"Progress: {crawled_count} pages crawled, {self.recipes_saved} recipes saved"
                )

            time.sleep(random.uniform(2, 5))

        self.driver.quit()
        self.save_checkpoint()
        self.logger.info("Crawling completed!")
        self.logger.info(f"Total pages visited: {len(self.visited_urls)}")
        self.logger.info(f"Total recipes saved: {self.recipes_saved}")


def main():
    parser = argparse.ArgumentParser(description="Crawler")
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Extract and compare recipe URLs from sitemap.xml",
    )
    args = parser.parse_args()

    if args.xml:
        robots_url = "https://foodnetwork.co.uk/robots.txt"
        try:
            resp = requests.get(robots_url, timeout=10)
            resp.raise_for_status()
            lines = resp.text.splitlines()
            sitemap_url = None
            for line in lines:
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    break
        except Exception as e:
            print(f"Failed to fetch robots.txt: {e}")
            return

        try:
            sitemap_resp = requests.get(sitemap_url, timeout=15)
            sitemap_resp.raise_for_status()
            root = ET.fromstring(sitemap_resp.content)
        except Exception as e:
            print(f"Failed to fetch or parse sitemap.xml: {e}")
            return

        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = set()

        if root.tag.endswith("sitemapindex"):
            sitemap_locs = [
                elem.find("ns:loc", ns).text
                for elem in root.findall("ns:sitemap", ns)
                if elem.find("ns:loc", ns) is not None
            ]
            for sm_url in sitemap_locs:
                try:
                    sm_resp = requests.get(sm_url, timeout=15)
                    sm_resp.raise_for_status()
                    sm_root = ET.fromstring(sm_resp.content)
                    for url_elem in sm_root.findall(".//ns:url", ns):
                        loc = url_elem.find("ns:loc", ns)
                        if loc is not None and "/recipes/" in loc.text:
                            path_parts = loc.text.split("/")
                            if len(path_parts) >= 5 and path_parts[3] == "recipes":
                                urls.add(loc.text.strip())
                except Exception as e:
                    print(f"Failed to fetch/parse child sitemap {sm_url}: {e}")
        else:
            for url_elem in root.findall(".//ns:url", ns):
                loc = url_elem.find("ns:loc", ns)
                if loc is not None and "/recipes/" in loc.text:
                    path_parts = loc.text.split("/")
                    if len(path_parts) >= 5 and path_parts[3] == "recipes":
                        urls.add(loc.text.strip())

        print(f"Found {len(urls)} recipe URLs in all sitemaps.")

        html_dir = config.RAW_HTML_DIR
        html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]

        html_urls = set()
        for filename in html_files:
            url = filename.replace(".html", "")
            url = url.replace("_", "/")
            if not url.startswith("http"):
                url = "https://" + url
            html_urls.add(url)

        in_sitemap_not_downloaded = urls - html_urls
        downloaded_not_in_sitemap = html_urls - urls

        print(
            f"\nRecipes in sitemap.xml but not downloaded ({len(in_sitemap_not_downloaded)}):"
        )

        if downloaded_not_in_sitemap:
            print(
                f"\nDeleting {len(downloaded_not_in_sitemap)} HTML files not in sitemap..."
            )
            for url in sorted(downloaded_not_in_sitemap):
                parsed = urlparse(url)
                url_filename = re.sub(r"[^\w\-_.]", "_", parsed.path.strip("/"))
                domain_prefix = re.sub(r"[^\w\-_.]", "_", parsed.netloc)
                filename = f"{domain_prefix}_{url_filename}.html"
                filepath = os.path.join(html_dir, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")
                    except Exception as e:
                        print(f"Failed to delete {filepath}: {e}")

        print(
            f"\nRecipes downloaded but not in sitemap.xml ({len(downloaded_not_in_sitemap)}):"
        )
        print(f"\nTotal in sitemap: {len(urls)} | Downloaded: {len(html_urls)}")

        if in_sitemap_not_downloaded:
            crawler = Crawler()
            for url in sorted(in_sitemap_not_downloaded):
                try:
                    crawler.crawl_page(url)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            crawler.driver.quit()
            crawler.save_checkpoint()
            print("\nDownload of missing recipes complete.")
        return

    crawler = Crawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
