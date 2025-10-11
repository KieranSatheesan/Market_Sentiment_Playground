# scripts/build_universe.py (FIXED VERSION)
import argparse, os, re, unicodedata
import pandas as pd

# --- helpers ---
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")

# EXPANDED STOP WORDS - CRITICAL FIX
FINANCE_STOPWORDS = {
    # Legal/Corporate
    "inc", "inc.", "corp", "corp.", "co", "co.", "ltd", "ltd.", "plc", "nv", "n.v.",
    "sa", "s.a.", "ag", "se", "ab", "oyj", "spa", "group", "holdings", "holding",
    "company", "the", "class", "series", "adr", "ads", "pte", "llc", "lp", "l.p.",
    "common", "stock", "stocks", "share", "shares", "corporation", 
    
    # Financial Instruments
    "preferred", "units", "unit", "warrant", "warrants", "rights", "ordinary",
    "depositary", "depository", "american", "trust", "fund", "income",
    
    # Generic Business Terms
    "technologies", "technology", "platforms", "systems", "solutions", 
    "international", "global", "bancorp", "financial", "capital", "properties", 
    "energy", "pharmaceuticals", "resources", "industries", "industry",
    
    # FINANCE-SPECIFIC GENERIC WORDS (CRITICAL ADDITION)
    "market", "trading", "services", "electronic", "income", "begin", "open",
    "legal", "name", "under", "lottery", "july", "multi", "acquisition",
    "management", "asset", "assets", "realty", "reit", "infrastructure",
    "parts", "auto", "battery", "brands", "lab", "labs", "laboratories",
    "healthcare", "medical", "therapeutics", "biologics", "biotech", "pharma",
    "mining", "gold", "silver", "metals", "entertainment", "gaming", "sports",
    "hotels", "resorts", "restaurants", "food", "ingredients", "consumer",
    "retail", "wholesale", "manufacturing", "industrial", "logistics",
    "transportation", "aviation", "shipping", "railroad", "utilities",
    "telecommunications", "media", "broadcasting", "publishing", "advertising",
    "consulting", "professional", "education", "training", "development",
    "research", "innovation", "venture", "private", "public", "equity",
    "debt", "credit", "loan", "mortgage", "insurance", "investment",
    "portfolio", "wealth", "asset", "banking", "payment", "processing",
    "digital", "online", "internet", "software", "hardware", "cloud",
    "data", "analytics", "intelligence", "security", "network", "wireless",
    "semiconductor", "device", "equipment", "material", "chemical", "steel",
    "oil", "gas", "petroleum", "renewable", "solar", "wind", "nuclear",
    "water", "waste", "environmental", "sustainability", "clean", "green"
}

def norm_ascii_lower(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())

def basic_aliases(name: str) -> list[str]:
    """
    FIXED: Generate meaningful aliases, not generic word soup
    """
    base = norm_ascii_lower(name)
    if not base:
        return []

    # Remove punctuation and split
    cleaned = re.sub(r"[&.,()\-/]", " ", base)
    tokens = [t for t in cleaned.split() if t and t not in FINANCE_STOPWORDS]

    aliases = set()
    
    if tokens:
        # STRICTER single-word aliases - only keep distinctive words
        for token in tokens:
            # Only keep single words that are likely to be distinctive
            if (len(token) >= 5 and  # Longer words are more distinctive
                token not in FINANCE_STOPWORDS and
                not token.isnumeric() and  # Skip numbers
                not re.search(r'\d', token)):  # Skip alphanumeric
                aliases.add(token)
        
        # Multi-word aliases - much more reliable
        if len(tokens) >= 2:
            # Company name without corporate cruft
            aliases.add(" ".join(tokens))
            
            # First 2-3 words (often the actual company name)
            if len(tokens) >= 2:
                aliases.add(" ".join(tokens[:2]))
            if len(tokens) >= 3:
                aliases.add(" ".join(tokens[:3]))
                
            # Last 2 words (sometimes distinctive)
            if len(tokens) >= 2:
                aliases.add(" ".join(tokens[-2:]))

    return sorted(a for a in aliases if len(a) >= 4)  # Minimum length increased

# ... rest of the code remains the same ...

def main():
    # ... existing main code ...
    
    print("Generating aliases...")
    uni["aliases_auto"] = uni["company_name"].apply(lambda s: ";".join(basic_aliases(s)))

    # DEBUG: Show what aliases are being generated for problematic tickers
    print("\n🔍 DEBUG: Alias generation for problematic tickers:")
    problematic = ['STBA', 'BRSL', 'HEPS', 'JMM', 'AACB']
    for ticker in problematic:
        if ticker in uni['ticker'].values:
            row = uni[uni['ticker'] == ticker].iloc[0]
            aliases = row['aliases_auto']
            print(f"  {ticker} ({row['company_name']}): {aliases}")
    
    # ... rest of main code ...