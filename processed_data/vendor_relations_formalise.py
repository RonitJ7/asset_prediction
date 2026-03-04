import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

# Load environment variables (Ensure GEMINI_API_KEY is in your .env file)

# --- Configuration ---
INPUT_CSV = "daily_returns.csv"
INPUT_JSON = "nifty100_vendor_relations.json"
OUTPUT_FILE = "nifty100_vendor_relations_cleaned.json"
MODEL_NAME = "gemini-2.5-flash" # Flash is faster/cheaper for high volume tasks

# Configure Gemini
API_KEY = input("Enter your Gemini API Key: ")
client = genai.Client(api_key=API_KEY)

# --- System Prompt ---
# This prompt enforces a strict JSON schema and defines the relationships we want.
SYSTEM_PROMPT = """
You are a financial data cleaning assistant. 
Your goal is to map company names to their official Nifty 100 tickers to create a clean knowledge graph.

**Input:**
1. A JSON object representing a company and its relationships (suppliers, clients, competitors).
2. A list of valid Nifty 100 tickers.

**Instructions:**
1. Analyze the lists in `relationships` (`suppliers`, `clients`, `competitors`).
2. For each company name found in these lists, check if it corresponds to one of the provided Nifty 100 tickers.
3. **Replace** the company name with the **exact ticker string** from the provided list.
4. **Filter**: If a company name does NOT match any ticker in the provided list, **remove** it from the relationship list. We ONLY want edges where both the source and target are in the Nifty 100.
5. Ensure the `ticker` field in the root object remains unchanged.
6. Return the cleaned JSON object.

**Output Format:**
Return ONLY the valid JSON object. Do not include markdown formatting.
"""

def get_nifty_tickers(csv_path):
    """Extracts stock tickers from the daily_returns CSV columns."""
    try:
        df = pd.read_csv(csv_path)
        # Filter out 'date' and any empty index columns like 'Unnamed: 0'
        stocks = [col for col in df.columns if col.lower() != 'date' and not col.startswith('Unnamed')]
        return stocks
    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")
        return []

def clean_record(record, valid_tickers):
    """Asks Gemini to map names in the record to valid tickers."""
    ticker = record.get('ticker')
    
    # Construct the user prompt with context
    user_content = f"""
    **Valid Nifty 100 Tickers:**
    {json.dumps(valid_tickers)}

    **Record to Clean:**
    {json.dumps(record)}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        # Return original record if cleaning fails, to avoid data loss
        return record

def main():
    # 1. Load Tickers
    print(f"Loading tickers from {INPUT_CSV}...")
    tickers = get_nifty_tickers(INPUT_CSV)
    if not tickers:
        print("No tickers found. Exiting.")
        return
    print(f"Loaded {len(tickers)} valid Nifty 100 tickers.")

    # 2. Load Existing JSON
    print(f"Loading relations from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Input JSON file not found at {INPUT_JSON}")
        return

    cleaned_data = []
    print(f"Starting cleaning process for {len(data)} records...")
    
    # 3. Process Records
    for record in tqdm(data):
        cleaned = clean_record(record, tickers)
        cleaned_data.append(cleaned)
        # Rate limiting
        time.sleep(1.0) 

    # 4. Save Output
    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()