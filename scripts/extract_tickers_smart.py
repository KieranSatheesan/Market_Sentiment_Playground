# scripts/extract_tickers_smart.py
import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------- spaCy for NLP ----------
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# =====================
#       UTILITIES
# =====================

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def norm_ascii_lower(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())

# =====================
#   SMART TICKER EXTRACTOR
# =====================

class SmartTickerExtractor:
    def __init__(self, universe_csv_path: str):
        self.valid_tickers = set()
        self.company_to_ticker = {}
        self.ticker_to_company = {}
        
        self.load_universe(universe_csv_path)
        
        # Load spaCy if available
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded for NLP extraction")
            except Exception:
                print("⚠️  Could not load spaCy model, proceeding without NLP")
                self.nlp = None
    
    def load_universe(self, csv_path: str):
        """Load valid tickers and build company->ticker mapping"""
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            print(f"📊 Loaded {len(df)} rows from universe CSV")
        except Exception as e:
            raise ValueError(f"Failed to load {csv_path}: {e}")
        
        if "ticker" not in df.columns:
            raise ValueError("CSV must contain 'ticker' column")
        
        company_col = "company_name" if "company_name" in df.columns else None
        
        for _, row in df.iterrows():
            try:
                ticker = str(row["ticker"]).strip().upper()
                if not ticker or len(ticker) > 10:
                    continue
                    
                self.valid_tickers.add(ticker)
                
                # Build company name mapping
                if company_col and pd.notna(row.get(company_col)):
                    company = str(row[company_col]).strip()
                    if company:
                        clean_company = self.clean_company_name(company)
                        if clean_company:
                            self.company_to_ticker[clean_company] = ticker
                            self.ticker_to_company[ticker] = clean_company
                            
            except Exception as e:
                print(f"⚠️  Skipping malformed row: {e}")
                continue
        
        print(f"✅ Built mapping: {len(self.valid_tickers)} tickers, {len(self.company_to_ticker)} company mappings")
    
    def clean_company_name(self, name: str) -> str:
        """Extract meaningful company name by removing corporate suffixes"""
        if not name:
            return ""
            
        # Common corporate suffixes to remove
        suffixes = {
            'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 
            'company', 'co', 'plc', 'sa', 'ag', 'group', 'holdings', 'holding',
            'llc', 'lp', 'nv', 'ab', 'se', 'adr', 'ordinary shares'
        }
        
        # Also remove these common words that don't help identification
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        name_lower = name.lower().strip()
        
        # Remove content in parentheses
        name_lower = re.sub(r'\([^)]*\)', '', name_lower)
        
        # Split and filter words
        words = name_lower.split()
        meaningful_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)  # Remove punctuation
            if (clean_word and 
                clean_word not in suffixes and 
                clean_word not in stop_words and
                len(clean_word) > 1):
                meaningful_words.append(clean_word)
        
        # Return the first 2-3 meaningful words (usually the core company name)
        if len(meaningful_words) >= 3:
            return ' '.join(meaningful_words[:3])
        elif meaningful_words:
            return ' '.join(meaningful_words)
        else:
            return name_lower  # Fallback to original
    
    def has_financial_context(self, text: str) -> bool:
        """Check if text has financial discussion context to reduce false positives"""
        if not text:
            return False
            
        finance_keywords = {
            'stock', 'stocks', 'price', 'prices', 'earnings', 'dividend', 'dividends',
            'buy', 'buying', 'sell', 'selling', 'long', 'short', 'position', 'positions',
            'portfolio', 'investment', 'invest', 'investing', 'investor', 'investors',
            'trading', 'trade', 'trades', 'market', 'markets', 'volume', 'shares',
            'share', 'valuation', 'target', 'pt', 'guidance', 'quarter', 'q1', 'q2',
            'q3', 'q4', 'eps', 'revenue', 'call', 'put', 'options', 'option',
            'downgrade', 'upgrade', 'analyst', 'analysts', 'coverage', 'initiate',
            'sector', 'industry', 'financial', 'finance'
        }
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in finance_keywords if keyword in text_lower)
        
        # Consider it financial context if at least 2 finance keywords are present
        return keyword_count >= 2
    
    def extract_cashtags(self, text: str) -> Set[str]:
        """Extract $TICKER patterns - most reliable signal"""
        if not text:
            return set()
        
        # Pattern for $TICKER, handles $AAPL, $TSLA, $BRK.B, $ORSTED.CO etc.
        pattern = r'\$([A-Za-z][A-Za-z0-9.\-]{1,8})\b'
        cashtags = set(re.findall(pattern, text.upper()))
        
        # Filter to valid tickers
        valid_cashtags = set()
        for ticker in cashtags:
            # Allow tickers with dots (international) even if not in our main list
            if '.' in ticker:
                valid_cashtags.add(ticker)
            elif ticker in self.valid_tickers:
                valid_cashtags.add(ticker)
        
        return valid_cashtags
    
    def extract_parentheses_tickers(self, text: str) -> Set[str]:
        """Extract (TICKER) or [TICKER] patterns with context validation"""
        if not text:
            return set()
        
        # Pattern for (TICKER) or [TICKER]
        pattern = r'[\(\[]([A-Z][A-Z0-9.\-]{1,8})[\)\]]'
        paren_tickers = set(re.findall(pattern, text.upper()))
        
        # Only include if we have financial context in the surrounding text
        if not self.has_financial_context(text):
            return set()
        
        # Filter to valid tickers
        valid_tickers = set()
        for ticker in paren_tickers:
            if '.' in ticker or ticker in self.valid_tickers:
                valid_tickers.add(ticker)
        
        return valid_tickers
    
    def extract_company_mentions(self, text: str) -> Set[str]:
        """Use NLP to find company mentions and map to tickers"""
        if not text or not self.nlp:
            return set()
        
        # Only process if there's financial context
        if not self.has_financial_context(text):
            return set()
        
        try:
            doc = self.nlp(text)
            found_tickers = set()
            
            for ent in doc.ents:
                if ent.label_ in ["ORG"]:  # Organization entities
                    company_name = self.clean_company_name(ent.text)
                    if company_name and company_name in self.company_to_ticker:
                        ticker = self.company_to_ticker[company_name]
                        found_tickers.add(ticker)
            
            return found_tickers
        except Exception as e:
            print(f"⚠️  NLP processing error: {e}")
            return set()
    
    def extract_tickers(self, text: str) -> Tuple[Set[str], Dict[str, List[str]]]:
        """
        Main extraction method using hybrid approach
        Returns:
            tickers: set of found tickers
            details: match type breakdown for debugging
        """
        if not text:
            return set(), {}
        
        # Track where each ticker was found
        details = {
            "cashtag": [],
            "paren": [], 
            "company_mention": []
        }
        
        all_tickers = set()
        
        # 1. Cash tags (most reliable - always include)
        cashtags = self.extract_cashtags(text)
        all_tickers.update(cashtags)
        details["cashtag"] = sorted(cashtags)
        
        # 2. Parentheses with context check
        paren_tickers = self.extract_parentheses_tickers(text)
        all_tickers.update(paren_tickers)
        details["paren"] = sorted(paren_tickers)
        
        # 3. Company mentions via NLP (most conservative)
        company_tickers = self.extract_company_mentions(text)
        all_tickers.update(company_tickers)
        details["company_mention"] = sorted(company_tickers)
        
        return all_tickers, details

# =====================
#   SCHEMAS (same as before)
# =====================

ENRICH_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("created_utc", pa.int64()),
    ("subreddit", pa.string()),
    ("title", pa.string()),
    ("selftext", pa.string()),
    ("score", pa.int32()),
    ("num_comments", pa.int32()),
    ("author", pa.string()),
    ("permalink", pa.string()),
    ("url", pa.string()),
    ("over_18", pa.bool_()),
    ("tickers", pa.list_(pa.string())),
])

EXPLODED_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("created_utc", pa.int64()),
    ("subreddit", pa.string()),
    ("author", pa.string()),
    ("ticker", pa.string()),
    ("from_title", pa.bool_()),
    ("from_body", pa.bool_()),
    ("match_types", pa.list_(pa.string())),
])

# =====================
#   PROCESSING FUNCTIONS
# =====================

def day_paths(clean_root: Path) -> List[Tuple[str, List[Path]]]:
    """Find cleaned input files"""
    days = []
    for d in sorted(clean_root.glob("*")):
        if d.is_dir():
            parts = list(d.glob("submissions_clean.parquet"))
            if not parts:
                parts = list(d.glob("submissions_clean_*.parquet"))
            if parts:
                days.append((d.name, parts))
    return days

def write_parquet(table: pa.Table, out_path: Path, schema: Optional[pa.Schema] = None):
    """Write parquet file with schema enforcement"""
    ensure_dir(out_path.parent)
    if schema is not None and table.schema != schema:
        table = table.cast(schema, safe=False)
    pq.write_table(table, out_path, compression="zstd")

def process_day(extractor: SmartTickerExtractor, day: str, input_files: List[Path]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Process a single day's data"""
    dfs = []
    for p in input_files:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"❌ Error reading {p}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(), []
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Ensure expected columns
    for col in ["id", "created_utc", "subreddit", "title", "selftext", "score", 
                "num_comments", "author", "permalink", "url", "over_18"]:
        if col not in df.columns:
            df[col] = "" if col in ["title", "selftext", "subreddit", "author", "permalink", "url"] else 0
    
    tickers_col = []
    exploded_rows = []
    
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        body = str(row.get("selftext", "") or "")
        full_text = f"{title} {body}".strip()
        
        # Extract tickers using the smart hybrid approach
        tickers, details = extractor.extract_tickers(full_text)
        tickers_list = sorted(tickers)
        tickers_col.append(tickers_list)
        
        # Build exploded rows for detailed analysis
        if tickers_list:
            for ticker in tickers_list:
                match_types = []
                if ticker in details["cashtag"]:
                    match_types.append("cashtag")
                if ticker in details["paren"]:
                    match_types.append("paren") 
                if ticker in details["company_mention"]:
                    match_types.append("company_mention")
                
                # Determine source
                title_lower = title.lower()
                body_lower = body.lower()
                from_title = any([
                    f"${ticker}".lower() in title_lower,
                    f"({ticker})".lower() in title_lower,
                    ticker.lower() in title_lower
                ])
                from_body = any([
                    f"${ticker}".lower() in body_lower,
                    f"({ticker})".lower() in body_lower,
                    ticker.lower() in body_lower
                ])
                
                exploded_rows.append({
                    "id": str(row["id"]),
                    "created_utc": int(pd.to_numeric(row["created_utc"], errors="coerce") or 0),
                    "subreddit": str(row.get("subreddit", "")),
                    "author": str(row.get("author", "")),
                    "ticker": ticker,
                    "from_title": from_title,
                    "from_body": from_body,
                    "match_types": match_types,
                })
    
    df["tickers"] = tickers_col
    return df, exploded_rows

# =====================
#        MAIN
# =====================

def main():
    ap = argparse.ArgumentParser(description="Extract tickers using smart hybrid approach")
    ap.add_argument("--clean_root", required=True, help="cleaned root: data/RedditDumps/cleaned")
    ap.add_argument("--out_enriched_root", required=True, help="output root for enriched: data/derived/submissions_with_tickers")
    ap.add_argument("--out_exploded_root", required=True, help="output root for exploded: data/derived/submission_tickers")
    ap.add_argument("--universe_csv", required=True, help="ref/ticker_universe.csv")
    ap.add_argument("--max_days", type=int, default=None, help="Process at most N days (for smoke tests)")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_enriched_root = Path(args.out_enriched_root)
    out_exploded_root = Path(args.out_exploded_root)

    # Initialize the smart extractor
    try:
        extractor = SmartTickerExtractor(args.universe_csv)
    except Exception as e:
        print(f"❌ Failed to initialize ticker extractor: {e}")
        return

    worklist = day_paths(clean_root)
    if args.max_days is not None:
        worklist = worklist[:args.max_days]

    if not worklist:
        print("[INFO] No cleaned day files found.")
        return

    for day, parts in worklist:
        print(f"\n[DAY] {day} | files={len(parts)}")
        
        # Process the day
        enriched_df, exploded_rows = process_day(extractor, day, parts)
        
        if enriched_df.empty:
            print("  - no rows processed")
            continue

        # Write enriched file
        enriched_df["over_18"] = enriched_df.get("over_18", False).fillna(False).astype(bool)
        enriched_df["score"] = pd.to_numeric(enriched_df["score"], errors="coerce").fillna(0).astype('int32')
        enriched_df["num_comments"] = pd.to_numeric(enriched_df["num_comments"], errors="coerce").fillna(0).astype('int32')
        enriched_df["created_utc"] = pd.to_numeric(enriched_df["created_utc"], errors="coerce").fillna(0).astype('int64')

        enriched_tbl = pa.Table.from_pandas(enriched_df, schema=ENRICH_SCHEMA, preserve_index=False)
        out_enriched_path = out_enriched_root / day / "submissions_with_tickers.parquet"
        write_parquet(enriched_tbl, out_enriched_path, schema=ENRICH_SCHEMA)
        print(f"  - wrote enriched: {out_enriched_path} ({enriched_tbl.num_rows} rows)")

        # Write exploded file
        if exploded_rows:
            exp_tbl = pa.Table.from_pylist(exploded_rows, schema=EXPLODED_SCHEMA)
            out_exploded_path = out_exploded_root / day / "submission_tickers.parquet"
            write_parquet(exp_tbl, out_exploded_path, schema=EXPLODED_SCHEMA)
            print(f"  - wrote exploded: {out_exploded_path} ({exp_tbl.num_rows} rows)")
        else:
            print("  - no exploded rows (no tickers found)")

if __name__ == "__main__":
    main()