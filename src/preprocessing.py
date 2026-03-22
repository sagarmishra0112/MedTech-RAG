import os
import json
import re

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "parsed_output")    
FULLTEXT_PATH = os.path.join(DATA_DIR, "fulltext.txt")
TABLES_PATH = os.path.join(DATA_DIR, "tables.json")
CLEAN_TEXT_PATH = os.path.join(DATA_DIR, "clean_text.txt")
PROCESSED_TABLES_PATH = os.path.join(DATA_DIR, "processed_tables.md")

# V1 Rules
EXCLUDE_PAGES = {11, 32} # 11 is a UI diagram, 32 has circuit schematic labels

def load_data():
    with open(FULLTEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return text, tables
 
def clean_text(text):
    print("Cleaning unstructured text...")
    
    # 1. Remove known headers and footers using Regex
    # Match phrases optionally followed by a newline
    text = re.sub(r'Allengers 100 Installation/Service Manual\n?', '', text)
    text = re.sub(r'AX 100 SM\n?', '', text)
    # Target "Page X" specifically at the end of lines or on its own line
    text = re.sub(r'^Page [a-zA-Z0-9]+\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'Page [a-zA-Z0-9]+\n?', '', text, flags=re.IGNORECASE)
    
    # Strip Company repeating header as well
    text = re.sub(r'AN ISO 9001:2008 & 13485:2003 COMPANY\n?', '', text)
    
    # 2. Normalize whitespace
    # Replace any sequence of 3 or more newlines with exactly 2 newlines 
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. Crop Starting Noise (TOC, Approvals, etc)
    # Split exactly at the text "1. SYSTEM OVERVIEW" and keep everything after
    split_keyword = "1. SYSTEM OVERVIEW"
    if split_keyword in text:
        # split(..., 1) splits only on the FIRST occurrence, giving us [before, after]
        text_parts = text.split(split_keyword, 1)
        # Re-attach the keyword so we don't lose the section header!
        text = split_keyword + text_parts[1]
    
    return text.strip()

def process_row(row_str):
    """Helper to convert the stringified list representation back to a normal list.
    Example: "['F1', 'Line Fuse']" -> ['F1', 'Line Fuse']
    """
    if isinstance(row_str, list):
        return [str(c) if c is not None else "" for c in row_str]
        
    try:
        # Safely convert string representation of list back to list
        import ast
        return ast.literal_eval(row_str)
    except:
        return []

def forward_fill_page_12(rows): 
    """Specific V1 fix for Page 12 Fuses Table"""
    cleaned_rows = []
    last_location = ""
    for row in rows:
        cells = process_row(row)
        
        # Strip exact empty columns LlamaParse threw in
        clean_cells = [c.strip() for c in cells if c.strip() != '']
        
        # Is this a fuse row? (Starts with F1, F2...)
        if len(clean_cells) > 0 and clean_cells[0].startswith('F'):
            if len(clean_cells) >= 4:
                # P12 format: ['F1', 'Line Fuse', '25 Amps.', 'On Rear of Control<br/>Panel']
                last_location = clean_cells[3]
            elif len(clean_cells) == 3 and last_location:
                # F2 row is missing the 4th element: ['F2', 'Filament Supply', '4 Amps.']
                # Forward fill the location!
                clean_cells.append(last_location)
        
        cleaned_rows.append(clean_cells)
    return cleaned_rows

def make_markdown_table(rows):
    if not rows:
        return ""
    
    # Use the first row to determine column count
    num_cols = len(rows[0])
    
    md = []
    # Header Row
    md.append("| " + " | ".join(rows[0]) + " |")
    # Separator Row
    md.append("|" + "---|"*num_cols)
    
    # Data Rows
    for row in rows[1:]:
        # Pad row to num_cols if it's too short
        padded_row = row + [""] * (num_cols - len(row))
        # Truncate if too long
        padded_row = padded_row[:num_cols]
        
        # Clean any line breaks inside cells (Markdown tables don't support \n)
        clean_row = [str(c).replace('\n', ' ').replace('<br/>', ' ') for c in padded_row]
        md.append("| " + " | ".join(clean_row) + " |")
    
    return "\n".join(md)

def process_tables(tables_data):
    print(f"Processing {len(tables_data)} tables...")
    markdown_output = []
    
    for table in tables_data:
        page = table.get("page", 0)
        
        # 1. Exclusion Rule
        if page in EXCLUDE_PAGES:
            print(f" ⏭️ Skipping fake table on Page {page}")
            continue
            
        raw_rows = table.get("rows", [])
        
        # 2. Forward Fill Rule
        if page == 12:
            processed_rows = forward_fill_page_12(raw_rows)
        else:
            # Normal processing for other pages
            processed_rows = [process_row(r) for r in raw_rows]
            # Clean up those weird empty columns ('') for clean markdown
            processed_rows = [[c.strip() for c in row if c.strip() != ''] for row in processed_rows]
        
        # Filter empty or 1-line "tables"
        if not processed_rows or len(processed_rows) <= 1:
             continue
             
        # Generate clean Markdown string
        md_table = make_markdown_table(processed_rows)
        
        # Add metadata/framing for RAG
        markdown_output.append(f"### Data Table - Source: Page {page}")
        markdown_output.append(md_table)
        markdown_output.append("\n")
        
    return "\n".join(markdown_output)


def main():
    print("🚀 Starting Preprocessing V1 Pipeline...")
    
    if not os.path.exists(FULLTEXT_PATH) or not os.path.exists(TABLES_PATH):
        print(f"❌ Error: Could not find ingestion output at {DATA_DIR}")
        return
        
    text, tables = load_data()
    
    # Execute Pipeline
    clean_txt = clean_text(text)
    processed_md = process_tables(tables)
    
    # Save structured outputs
    with open(CLEAN_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(clean_txt)
    print(f"✅ Saved cleaned full text -> {os.path.basename(CLEAN_TEXT_PATH)}")
    
    with open(PROCESSED_TABLES_PATH, "w", encoding="utf-8") as f:
        f.write(processed_md)
    print(f"✅ Saved formatted Markdown tables -> {os.path.basename(PROCESSED_TABLES_PATH)}")
    
    print("Done! Data is ready for RAG Chunking.")

if __name__ == "__main__":
    main()