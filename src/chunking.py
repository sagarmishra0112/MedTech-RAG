import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "parsed_output")
CLEAN_TEXT_PATH = os.path.join(DATA_DIR, "clean_text.txt")
PROCESSED_TABLES_PATH = os.path.join(DATA_DIR, "processed_tables.md")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")

def ensure_dirs():
    if not os.path.exists(CHUNKS_DIR):
        os.makedirs(CHUNKS_DIR)

def chunk_text():
    print("Chunking unstructured text...")
    if not os.path.exists(CLEAN_TEXT_PATH):
        print(f"❌ Error: Could not find {CLEAN_TEXT_PATH}")
        return []

    with open(CLEAN_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Create the text splitter
    # We use a chunk size of 2000 characters (~500 tokens) to capture full paragraphs
    # We use 300 characters (~75 tokens) of overlap to prevent breaking sentences in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Prioritize splitting at paragraphs
    )

    chunks = text_splitter.split_text(text)
    print(f"✅ Generated {len(chunks)} text chunks.")
    return chunks

def chunk_tables():
    print("Chunking markdown tables...")
    if not os.path.exists(PROCESSED_TABLES_PATH):
        print(f"❌ Error: Could not find {PROCESSED_TABLES_PATH}")
        return []

    with open(PROCESSED_TABLES_PATH, "r", encoding="utf-8") as f:
        tables_text = f.read()

    # Split exclusively by our custom header string
    # This guarantees that tables are NEVER split in half, preserving all rows
    raw_chunks = tables_text.split("### Data Table - Source: Page ")
    
    table_chunks = []
    # Skip any empty chunks (like the one before the first header)
    for raw in raw_chunks: 
        if not raw.strip():
            continue
        # Re-attach the metadata string that was consumed by the split
        text_chunk = f"### Data Table - Source: Page {raw.strip()}"
        table_chunks.append(text_chunk)

    print(f"✅ Generated {len(table_chunks)} table chunks.")
    return table_chunks

def main(): 
    print("🚀 Starting Semantic Chunking Pipeline...")
    ensure_dirs()
    
    text_chunks = chunk_text()
    table_chunks = chunk_tables()
    
    total_chunks = len(text_chunks) + len(table_chunks)
    
    # Save all chunks to a single file for review
    all_chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.txt")
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        f.write(f"Total Chunks Generated: {total_chunks}\n")
        f.write(f"Text Chunks: {len(text_chunks)} | Table Chunks: {len(table_chunks)}\n")
        f.write("=========================================\n\n")
        
        f.write("--- ALL TEXT CHUNKS ---\n\n")
        for i, chunk in enumerate(text_chunks):
            f.write(f"--- [TEXT CHUNK {i+1}] ---\n")
            f.write(chunk)
            f.write("\n\n")
            
        f.write("=========================================\n")
        f.write("--- ALL TABLE CHUNKS ---\n\n")
        for i, chunk in enumerate(table_chunks):
             f.write(f"--- [TABLE CHUNK {i+1}] ---\n")
             f.write(chunk)
             f.write("\n\n")
        
    print(f"🎯 Total pieces of knowledge generated for the Vector Database: {total_chunks}")
    print(f"📄 View all output chunks here: data/parsed_output/chunks/all_chunks.txt")

if __name__ == "__main__":
    main()
