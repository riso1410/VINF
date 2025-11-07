#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPES_FILE="${PROJECT_ROOT}/data/scraped/recipes.jsonl"

if [[ -f "${RECIPES_FILE}" ]]; then
    echo "Recipes already exist at ${RECIPES_FILE}; skipping extraction."
else
    echo "Recipes JSONL missing. Running Spark recipe extraction..."
    python3 "${PROJECT_ROOT}/src/recipes_extraction_spark.py" >/dev/null
fi

echo 'Starting Spark Wikipedia mapper'
python3 "${PROJECT_ROOT}/src/wiki_mapper_spark.py" >/dev/null
