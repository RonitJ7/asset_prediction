import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

# Load environment variables (Ensure GEMINI_API_KEY is in your .env file)

# --- Configuration ---
INPUT_FILE = "daily_returns.csv"
OUTPUT_FILE = "nifty100_vendor_relations.json"
MODEL_NAME = "gemini-2.5-flash" # Flash is faster/cheaper for high volume tasks

# Configure Gemini
API_KEY = input("Enter your Gemini API Key: ")
client = genai.Client(api_key=API_KEY)

# --- System Prompt ---
# This prompt enforces a strict JSON schema and defines the relationships we want.
SYSTEM_PROMPT = """
You are a financial supply chain analyst. Your task is to identify the key business relationships for a specific Indian public company.

For the company provided, identify:
1. **Vendors**: Companies that provide raw materials, software, or services to the target company.
2. **Clients**: Major companies that buy products or services from the target company.
3. **Competitors**: Direct rivals in the same industry segments.

**Constraints:**
- Focus on major, publicly known relationships.
- If specific company names are not known, do not hallucinate. Return an empty list.
- Output MUST be valid JSON only. Do not include markdown formatting like ```json ... ```.

**Output Format:**
{
  "ticker": "The input ticker",
  "company_name": "Full name of the company",
  "relationships": {
    "suppliers": ["Company A", "Company B"],
    "clients": ["Company C", "Company D"],
    "competitors": ["Company E", "Company F"]
  },
  "explanation": "Your reasoning for this analysis. Describe your sources in words but do not include a URL in the explanation."
}
"""

def get_stock_list(csv_path):
    """Extracts stock tickers from the daily_returns CSV columns."""
    try:
        df = pd.read_csv(csv_path)
        # Assuming the first column is Date and the rest are tickers
        # If your CSV structure is different, adjust this line.
        stocks = [col for col in df.columns if col.lower() != 'date']
        return stocks
    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")
        return []

def query_gemini(ticker):
    """Queries Gemini for a specific ticker."""
    
    user_prompt = f"Analyze the supply chain and competitors for the Indian stock: {ticker} (Nifty 100)."
    
    try:
        # Updated generation call
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {
            "ticker": ticker,
            "error": str(e),
            "relationships": {"suppliers": [], "clients": [], "competitors": []},
            "explanation": "Failed to retrieve data."
        }
        time.sleep(30)

def main():
    # 1. Get the list of stocks
    print(f"Loading stocks from {INPUT_FILE}...")
    stocks = get_stock_list(INPUT_FILE)
    
    if not stocks:
        print("No stocks found. Exiting.")
        return

    print(f"Found {len(stocks)} stocks. Starting Gemini extraction...")
    start_input = input(f"Enter start index (1 to {len(stocks)-1}, default 1): ")
    start_index = int(start_input) if start_input.strip() else 1
    end_input = input(f"Enter end index (1 to {len(stocks)}, default {len(stocks)}): ")
    end_index = int(end_input) if end_input.strip() else len(stocks)
    stocks = stocks[start_index:end_index]

    print("Fetching available models...")
    try:
        # List models
        pager = client.models.list()
        for model in pager:
            # Filter for generateContent support which is needed for text generation
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
    # 2. Iterate through stocks and query Gemini
    # Using tqdm for a progress bar
    for ticker in tqdm(stocks):
        data = query_gemini(ticker)
        
        # Rate Limiting: Gemini has limits (e.g., 15 RPM on free tier). 
        # Sleep for 4 seconds to stay safe (15 requests/min = 1 req every 4s).
        # Adjust this based on your API tier.
        with open(OUTPUT_FILE, "a") as f:
            json.dump(data, f, indent=2)
        time.sleep(15) 
    

    print(f"Successfully saved vendor relations data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()