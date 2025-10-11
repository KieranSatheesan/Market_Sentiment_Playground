# scripts/clean_ticker_universe_simple.py
import pandas as pd
import re

def clean_company_name(name: str) -> str:
    """Extract only the meaningful part of company name"""
    if not name or pd.isna(name):
        return ""
    
    # Remove everything in parentheses
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove common legal suffixes
    suffixes = [
        'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
        'company', 'co', 'plc', 'sa', 'ag', 'group', 'holdings', 'holding',
        'common stock', 'class a', 'class b', 'class c', 'american depositary',
        'depositary shares', 'units', 'warrant', 'rights', 'ordinary shares',
        'notes', 'senior', 'preferred', 'series'
    ]
    
    words = name.lower().split()
    meaningful_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if (clean_word and 
            clean_word not in suffixes and
            len(clean_word) > 2):
            meaningful_words.append(clean_word)
    
    # Return first 2-3 words (the actual company name)
    if meaningful_words:
        return ' '.join(meaningful_words[:3])
    return ""

def create_clean_universe(input_csv, output_csv):
    """Create a clean universe using ONLY company names as aliases"""
    df = pd.read_csv(input_csv)
    
    print(f"📊 Original data: {len(df)} rows")
    
    # Create clean aliases from company names only
    df['aliases'] = df['company_name'].apply(clean_company_name)
    
    # Remove rows where we couldn't extract a meaningful name
    df = df[df['aliases'].str.len() > 0]
    
    print(f"📊 Cleaned data: {len(df)} rows")
    
    # Show examples
    print("\n🔍 Examples:")
    for _, row in df.head(8).iterrows():
        print(f"  {row['ticker']}: '{row['company_name']}' → '{row['aliases']}'")
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved to: {output_csv}")
    
    return df

if __name__ == "__main__":
    create_clean_universe(
        "ref/ticker_universe.csv", 
        "ref/ticker_universe_cleaned.csv"
    )