
import asyncio
import os
import re
import json
import httpx
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from collections import Counter
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

# pip install llama_cloud>=1.0
# pip install --upgrade llama-cloud

load_dotenv(override=True)  
api_key = os.getenv("llamacloud_key")

# Output directories (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SCREENSHOTS_DIR = os.path.join(DATA_DIR, "screenshots")
OUTPUT_DIR = os.path.join(DATA_DIR, "parsed_output")

def is_page_screenshot(image_name: str) -> bool:
    return re.match(r"^page_(\d+)\.jpg$", image_name) is not None

async def main():
    if not api_key:
        print("Error: llamacloud_key not found in environment variables.")
        return

    # Create output directories if they don't exist
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = AsyncLlamaCloud(api_key=api_key, timeout=600.0)

    # Upload and parse a document
    file_path = os.path.join(DATA_DIR, "Allengers_100.pdf")
    if not os.path.exists(file_path):
        print(f"Input File Missing: {file_path}")
        return

    print(f"Uploading {file_path}...")
    try:
        file_obj = await client.files.create(file=file_path, purpose="parse")
    except Exception as e:
        print(f"Error creating file: {e}")
        return

    if not file_obj:
        print("Input File Missing (upload failed?)")
        return

    print("Parsing document...")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        input_options={},
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": False,
                },
            },
            "images_to_save": ["screenshot"],
        },
        processing_options={
            "ignore": {
                "ignore_diagonal_text": True,
            },
            "ocr_parameters": {
                "languages": ["en"]
            }
        },
        expand=["text", "markdown", "items", "images_content_metadata"],
    )

    # --- PREVIEW ---
    if result.markdown.pages:
        print("First page markdown preview:")
        print(result.markdown.pages[0].markdown[:500])
    
    # --- TABLE EXTRACTION ---
    tables_data = []  # We'll save this to a JSON file
    for page in result.items.pages:
        for item in page.items:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")

            if item_type == "table":
                if isinstance(item, dict):
                    rows = item.get("rows")
                    bbox = item.get("b_box")
                else:
                    rows = getattr(item, "rows", None)
                    bbox = getattr(item, "b_box", None)

                if rows:
                    print(f"Table found on page {page.page_number} with {len(rows)} rows")
                    # Convert rows to a serializable format for saving
                    serializable_rows = []
                    for row in rows:
                        if isinstance(row, dict):
                            serializable_rows.append(row)
                        elif hasattr(row, '__dict__'):
                            serializable_rows.append(row.__dict__)
                        else:
                            serializable_rows.append(str(row))
                    tables_data.append({
                        "page": page.page_number,
                        "num_rows": len(rows),
                        "rows": serializable_rows
                    })

    # --- DOWNLOAD SCREENSHOTS (to data/screenshots/, skip duplicates) ---
    downloaded = set()
    for image in result.images_content_metadata.images:
        if image.presigned_url is None or not is_page_screenshot(image.filename):
            continue
        if image.filename in downloaded:
            continue  # Skip duplicate downloads
        downloaded.add(image.filename)

        save_path = os.path.join(SCREENSHOTS_DIR, image.filename)
        print(f"Downloading {image.filename}, {image.size_bytes} bytes")
        with open(save_path, "wb") as img_file:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image.presigned_url)
                img_file.write(response.content)

    # --- SAVE PARSED DATA TO FILES ---
    # 1. Save plain text (for preprocessing.py to use later)
    fulltext = "\n\n".join(page.text for page in result.text.pages)
    fulltext_path = os.path.join(OUTPUT_DIR, "fulltext.txt")
    with open(fulltext_path, "w", encoding="utf-8") as f:
        f.write(fulltext)
    print(f"\nSaved plain text → {fulltext_path} ({len(fulltext)} chars)")

    # 2. Save full markdown
    full_markdown = "\n\n".join(page.markdown for page in result.markdown.pages)
    markdown_path = os.path.join(OUTPUT_DIR, "full_markdown.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(f"Saved markdown   → {markdown_path} ({len(full_markdown)} chars)")

    # 3. Save table data as JSON
    tables_path = os.path.join(OUTPUT_DIR, "tables.json")
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(tables_data, f, indent=2, default=str)
    print(f"Saved tables     → {tables_path} ({len(tables_data)} tables)")

    print(f"\nAll done! Screenshots: {len(downloaded)} pages, Tables: {len(tables_data)}")

if __name__ == "__main__":
    asyncio.run(main())
