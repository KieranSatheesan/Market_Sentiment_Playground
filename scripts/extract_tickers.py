# scripts/extract_tickers.py
import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------- Optional spaCy (off by default) ----------
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
#   UNIVERSE LOADING (FIXED)
# =====================

def load_universe(path: str | Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    ROBUST VERSION: Handles missing columns and malformed data
    Returns:
      valid_tickers: set of tickers (e.g., 'AAPL', 'BRK.B', 'ORSTED.CO')
      alias2tickers: alias -> set({ticker, ...})
    """
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        print(f"📊 Loaded {len(df)} rows from {path}")
    except Exception as e:
        raise ValueError(f"Failed to load {path}: {e}")
    
    # REQUIRED: Must have ticker column
    if "ticker" not in df.columns:
        raise ValueError("CSV must contain 'ticker' column")
    
    # OPTIONAL: Handle missing columns gracefully
    company_col = "company_name" if "company_name" in df.columns else None
    aliases_col = "aliases" if "aliases" in df.columns else None
    exchange_col = "exchange" if "exchange" in df.columns else None

    valid_tickers = set()
    alias2tickers: Dict[str, Set[str]] = {}
    
    for _, r in df.iterrows():
        try:
            ticker = str(r["ticker"]).strip().upper()
            if not ticker or len(ticker) > 10:  # More lenient length check
                continue
                
            valid_tickers.add(ticker)
            
            # Build aliases from available columns
            raw_aliases = []
            
            # Company name as alias
            if company_col and pd.notna(r.get(company_col)):
                company = str(r[company_col]).strip()
                if company:
                    raw_aliases.append(company)
            
            # Explicit aliases column
            if aliases_col and pd.notna(r.get(aliases_col)):
                alias_str = str(r[aliases_col])
                if alias_str:
                    raw_aliases.extend(alias_str.split(";"))
            
            # Process each alias
            for a in raw_aliases:
                a_n = norm_ascii_lower(a)
                if not a_n:
                    continue
                
                # Basic guardrails: avoid super-short single words that cause noise
                if (len(a_n) < 4) and (" " not in a_n):
                    continue
                    
                # Drop very generic junk words
                if a_n in GENERIC_STOP_ALIASES:
                    continue
                    
                alias2tickers.setdefault(a_n, set()).add(ticker)
                
        except Exception as e:
            print(f"⚠️  Skipping malformed row for ticker {r.get('ticker', 'unknown')}: {e}")
            continue

    print(f"✅ Built dictionary: {len(valid_tickers)} valid tickers, {len(alias2tickers)} aliases")
    return valid_tickers, alias2tickers

# Very generic words to ignore as aliases (extend as needed)
GENERIC_STOP_ALIASES = {
    "group", "inc", "corp", "corporation", "company", "co", "limited", "ltd",
    "holdings", "holding", "plc", "sa", "nv", "ag", "as", "llc", "lp",
    "preferred", "units", "unit", "warrant", "warrants", "rights", "ordinary",
    "class", "series", "stock", "common", "shares", "share", "depositary",
    "depository", "american", "global", "international", "technologies",
    "technology", "resources", "financial", "financials", "pharma", "biotech",
    "energy", "mining", "gold", "silver", "metals", "industries", "industry",
    "asset", "assets", "trust", "realty", "reit", "infrastructure",
    # AAP "parts" problem and other false positives
    "parts", "auto", "battery", "management", "brands", "lab", "labs", "laboratories"
}

# Tickers that are common English words or single letters we'll treat conservatively
CONSERVATIVE_TICKERS = {
    # Single letters - these DEFINITELY appear everywhere
    "A", "B", "D", "F", "I", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Z",
    
    # Common words that appear constantly in normal text
    "AI", "ALL", "AM", "AN", "ANY", "ARE", "AS", "AT", "BE", "BIG", "BY", "CAN", "CAR", 
    "CEO", "CO", "COM", "COP", "DAY", "DO", "EAT", "FOR", "GO", "GOT", "HAS", "HE", "HER",
    "HI", "HIM", "HOME", "HOW", "HR", "IF", "IN", "IO", "IP", "IS", "IT", "JOB", "KEY",
    "LIFE", "LIKE", "LIVE", "LOVE", "LOW", "MAN", "MAY", "ME", "MEN", "MORE", "MOST", "MY",
    "NET", "NEW", "NO", "NOR", "NOT", "NOW", "OF", "OH", "OK", "ON", "ONE", "OR", "OUR",
    "OUT", "OWN", "PAY", "PC", "PE", "PI", "PM", "PR", "PSA", "RE", "RUN", "SAY", "SEE",
    "SHE", "SO", "SPOT", "SUN", "TALK", "TAXI", "TD", "THE", "TO", "TRUE", "TV", "TWO",
    "UK", "UP", "US", "USA", "USE", "VIA", "VP", "WAS", "WAY", "WE", "WELL", "WERE", "WHAT",
    "WHEN", "WHERE", "WHO", "WHY", "WILL", "WIN", "WITH", "WORK", "YOU", "YOUR",
    
    # Common abbreviations used in normal conversation
    "AA", "AB", "AC", "AD", "AG", "AGO", "AH", "AL", "AP", "AR", "AT", "BA", "BC", "BE",
    "CA", "CC", "CD", "CE", "CF", "CI", "CM", "CO", "CP", "CT", "CV", "DC", "DD", "DE",
    "DM", "DR", "EA", "ED", "EG", "EM", "ER", "ES", "ET", "EU", "EV", "EX", "FA", "FB",
    "FC", "FD", "FT", "GA", "GB", "GC", "GM", "GP", "HA", "HB", "HD", "HP", "IA", "IB",
    "IC", "ID", "IE", "IL", "IN", "IR", "IT", "IV", "JR", "LA", "LB", "LC", "MA", "MB",
    "MC", "MD", "ME", "MI", "MM", "MO", "MP", "MR", "MS", "MT", "NA", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PB", "PC", "PD", "PDT", "PE",
    "PH", "PM", "PR", "PS", "PT", "QA", "RA", "RC", "RE", "RI", "SA", "SC", "SD", "SE",
    "SO", "SP", "SR", "ST", "TA", "TB", "TC", "TD", "TN", "TX", "UT", "VA", "VT", "WA",
    "WI", "WV", "WY",
    
    # Reddit/internet common terms
    "AMA", "API", "APP", "BOT", "CEO", "DM", "FAQ", "GIF", "GPS", "HR", "IMO", "IT", "LOL",
    "OMG", "PDF", "PM", "PR", "TIL", "URL", "USB", "VP", "WTF", "WWW",
    
    # Financial terms used conversationally
    "BID", "BULL", "CALL", "CASH", "EARN", "GAIN", "LOAN", "LOSS", "PAY", "RATE", "SAVE",
    
    # Days/months
    "AM", "APR", "AUG", "DEC", "FEB", "FRI", "JAN", "JUL", "JUN", "MAR", "MAY", "MON",
    "NOV", "OCT", "PM", "SAT", "SEP", "SUN", "THU", "TUE", "WED",
    
    # Generic business terms that appear in normal discussion
    "AI", "AUTO", "BEST", "CARE", "COLD", "COOL", "COST", "DATA", "FAST", "FIT", "FREE",
    "GAIN", "GOLD", "GOOD", "HEAR", "HEAT", "HOPE", "HUNT", "NEXT", "NICE", "OPEN", "PEAK",
    "PLUS", "PURE", "RACE", "REAL", "ROCK", "SAFE", "SHOP", "SITE", "SNAP", "STAR", "STAY",
    "TEAM", "TECH", "TRIP", "TRUE", "TURN", "TYPE", "UNIT", "WARM", "WAVE", "WISE", "ZONE",
}

# Finance context keywords (if you want to make small tickers less strict)
FINANCE_KEYWORDS = {
    "stock", "stocks", "ticker", "price", "shares", "earnings", "dividend",
    "market", "volume", "buy", "sell", "long", "short", "pt", "guidance",
    "quarter", "q1", "q2", "q3", "q4", "forecast", "eps", "revenue",
    "valuation", "call", "skim", "downgrade", "upgrade", "analyst", "target"
}

# =====================
#   REGEX EXTRACTORS
# =====================

# $TSLA, $BRK.B, $ORSTED.CO
PAT_CASHTAG = re.compile(r'\$([A-Za-z][A-Za-z0-9.\-]{0,5})\b')

# (TSLA) or [TSLA]
PAT_PAREN = re.compile(r'[\(\[]([A-Z][A-Z0-9.\-]{0,5})[\)\]]')

# NYSE: TSLA, NASDAQ: AAPL, LSE: VOD.L, TSX: SU.TO
PAT_EXCHANGE = re.compile(r'\b(?:NYSE|NASDAQ|AMEX|LSE|TSX|ASX|CBOE|OTC):\s*([A-Z][A-Z0-9.\-]{0,6})\b')

# Uppercase token candidates (filtered later by valid_tickers)
PAT_UPPER = re.compile(r'\b([A-Z]{2,6}(?:\.[A-Z]{1,3})?)\b')

def extract_by_regex(text: str) -> Dict[str, List[str]]:
    """Return dict of match_type -> [tickers...] (raw, not validated)."""
    if not text:
        return {"cashtag": [], "paren": [], "exchange": [], "upper": []}
    cashtag = [m.group(1).upper() for m in PAT_CASHTAG.finditer(text)]
    paren = [m.group(1).upper() for m in PAT_PAREN.finditer(text)]
    exch  = [m.group(1).upper() for m in PAT_EXCHANGE.finditer(text)]
    upper = [m.group(1).upper() for m in PAT_UPPER.finditer(text)]
    return {"cashtag": cashtag, "paren": paren, "exchange": exch, "upper": upper}

# =====================
#  VALIDATION / MERGE
# =====================

def refine_upper_candidates(uppers: List[str], valid_tickers: Set[str], text_norm: str) -> List[str]:
    """
    Filter uppercase tokens to valid tickers with conservative rules.
    - Must be in valid_tickers
    - If in CONSERVATIVE_TICKERS or length <= 2, require presence of finance keywords
    """
    if not uppers:
        return []
    keep = []
    has_fin_kw = any(kw in text_norm for kw in FINANCE_KEYWORDS)
    for u in uppers:
        if u not in valid_tickers:
            continue
        if (u in CONSERVATIVE_TICKERS) or (len(u) <= 2):
            if not has_fin_kw:
                continue
        keep.append(u)
    return keep

def alias_hits(text_norm: str, alias2tickers: Dict[str, Set[str]]) -> Set[str]:
    """
    Naive substring alias matching over normalized lower-ascii text.
    Guardrails:
    - alias length >= 4 or multi-word already filtered when loading
    """
    hits: Set[str] = set()
    for alias, tics in alias2tickers.items():
        if alias and alias in text_norm:
            hits |= tics
    return hits

def extract_tickers_from_text(title: str, body: str,
                              valid_tickers: Set[str],
                              alias2tickers: Dict[str, Set[str]]) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Combine regex + aliases with safeguards.
    Returns:
      tickers: set of validated tickers
      details: match_type -> unique list
    """
    title = title or ""
    body = body or ""
    whole = f"{title} {body}".strip()

    reg = extract_by_regex(whole)
    text_norm = norm_ascii_lower(whole)

    # Validate UPPER candidates post-filter
    reg["upper"] = refine_upper_candidates(reg["upper"], valid_tickers, text_norm)

    # Accept cashtag/paren/exchange candidates only if valid tickers or "dot" style
    def keep_valid(xs: List[str]) -> List[str]:
        keep = []
        for x in xs:
            x_up = x.upper()
            if x_up in valid_tickers:
                keep.append(x_up)
            else:
                # permit exchange-specific dotted tickers not in NYSE/NASDAQ (e.g., ORSTED.CO)
                if "." in x_up:
                    keep.append(x_up)
        return list(dict.fromkeys(keep))  # unique, preserve order

    cashtags = keep_valid(reg["cashtag"])
    parens   = keep_valid(reg["paren"])
    exch     = keep_valid(reg["exchange"])
    uppers   = list(dict.fromkeys(reg["upper"]))

    # Alias substring hits → map to tickers
    alias_tics = alias_hits(text_norm, alias2tickers)

    # Merge
    merged = set(cashtags) | set(parens) | set(exch) | set(uppers) | set(alias_tics)

    # Conservative rule for single-letter/common-word tickers unless seen via strong signals
    strong = set(cashtags) | set(parens) | set(exch)
    final = set()
    for t in merged:
        if (t in strong) or (t not in CONSERVATIVE_TICKERS):
            final.add(t)
        else:
            # allow if finance keywords present AND length >= 2 (already filtered)
            pass

    details = {
        "cashtag": sorted(set(cashtags)),
        "paren":   sorted(set(parens)),
        "exchange":sorted(set(exch)),
        "upper":   sorted(set(uppers)),
        "alias":   sorted(set(alias_tics)),
    }
    return final, details

# =====================
#       IO LAYER
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
    ("match_types", pa.list_(pa.string())),  # e.g., ["cashtag","alias"]
])

def day_paths(clean_root: Path) -> List[Tuple[str, List[Path]]]:
    """
    Find cleaned inputs: .../cleaned/YYYY-MM-DD/submissions_clean.parquet
    """
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
    ensure_dir(out_path.parent)
    if schema is not None and table.schema != schema:
        # cast if columns align by name
        table = table.cast(schema, safe=False)
    pq.write_table(table, out_path, compression="zstd")

# =====================
#        MAIN
# =====================

def main():
    ap = argparse.ArgumentParser(description="Extract tickers from cleaned Reddit submissions.")
    ap.add_argument("--clean_root", required=True, help="cleaned root: data/RedditDumps/cleaned")
    ap.add_argument("--out_enriched_root", required=True, help="output root for enriched: data/derived/submissions_with_tickers")
    ap.add_argument("--out_exploded_root", required=True, help="output root for exploded: data/derived/submission_tickers")
    ap.add_argument("--universe_csv", required=True, help="ref/ticker_universe.csv")
    ap.add_argument("--use_spacy", action="store_true", help="(optional) use spaCy ORG NER to add extra alias matches")
    ap.add_argument("--max_days", type=int, default=None, help="Process at most N days (for smoke tests)")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_enriched_root = Path(args.out_enriched_root)
    out_exploded_root = Path(args.out_exploded_root)

    # Load universe with robust error handling
    try:
        valid_tickers, alias2tickers = load_universe(args.universe_csv)
    except Exception as e:
        print(f"❌ Failed to load ticker universe: {e}")
        return

    if args.use_spacy and not _SPACY_AVAILABLE:
        print("[WARN] spaCy not available; proceeding without it.")
        args.use_spacy = False

    nlp = None
    if args.use_spacy:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except Exception:
            print("[WARN] Could not load 'en_core_web_sm'; spaCy disabled.")
            args.use_spacy = False
            nlp = None

    worklist = day_paths(clean_root)
    if args.max_days is not None:
        worklist = worklist[: args.max_days]

    if not worklist:
        print("[INFO] No cleaned day files found.")
        return

    for day, parts in worklist:
        print(f"\n[DAY] {day} | files={len(parts)}")
        # read into one DF (days are usually small)
        dfs = []
        for p in parts:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception as e:
                print(f"❌ Error reading {p}: {e}")
                continue
                
        if not dfs:
            print("  - no rows")
            continue
        df = pd.concat(dfs, ignore_index=True)

        # Ensure expected columns exist
        for c in ["id","created_utc","subreddit","title","selftext","score","num_comments","author","permalink","url","over_18"]:
            if c not in df.columns:
                df[c] = np.nan

        tickers_col: List[List[str]] = []
        exploded_rows: List[Dict[str, object]] = []

        for _, row in df.iterrows():
            title = row.get("title", "") or ""
            body  = row.get("selftext", "") or ""
            subs  = row.get("subreddit", "") or ""

            # First pass: regex + alias
            tickers, details = extract_tickers_from_text(title, body, valid_tickers, alias2tickers)

            # Optional spaCy NER pass (light): add ORG matches by name->alias map
            if args.use_spacy and nlp and (title or body):
                try:
                    doc = nlp(title + " " + body)
                    org_names = [norm_ascii_lower(ent.text) for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT")]
                    extra = set()
                    for org in org_names:
                        if org in alias2tickers:
                            extra |= alias2tickers[org]
                    if extra:
                        tickers |= extra
                        details["alias"] = sorted(set(details["alias"]) | extra)
                except Exception as e:
                    print(f"⚠️  spaCy error: {e}")

            # Save tickers array to enriched
            tickers_list = sorted(tickers)
            tickers_col.append(tickers_list)

            # Build exploded rows
            if tickers_list:
                for t in tickers_list:
                    mt = []
                    for k in ("cashtag","paren","exchange","upper","alias"):
                        if t in details.get(k, []):
                            mt.append(k)
                    
                    # Determine where ticker was found
                    tit_low = title.lower()
                    bod_low = body.lower()
                    from_title = any([
                        f"${t}".lower() in tit_low,
                        f"({t})".lower() in tit_low,
                        t.lower() in tit_low
                    ])
                    from_body = any([
                        f"${t}".lower() in bod_low,
                        f"({t})".lower() in bod_low, 
                        t.lower() in bod_low
                    ])

                    exploded_rows.append({
                        "id": str(row["id"]),
                        "created_utc": int(pd.to_numeric(row["created_utc"], errors="coerce") or 0),
                        "subreddit": str(subs),
                        "author": str(row.get("author","") or ""),
                        "ticker": t,
                        "from_title": from_title,
                        "from_body": from_body,
                        "match_types": mt,
                    })

        # ----- Write enriched day -----
        enriched_df = df.copy()
        enriched_df["tickers"] = tickers_col

        # Cast to Arrow table (fixed schema)
        def _coerce_bool(x):
            if isinstance(x, pd.Series):
                return x.fillna(False).astype(bool)
            return bool(x) if pd.notna(x) else False

        enriched_df["over_18"] = _coerce_bool(enriched_df.get("over_18", False))
        enriched_df["score"] = pd.to_numeric(enriched_df["score"], errors="coerce").fillna(0).astype(np.int32)
        enriched_df["num_comments"] = pd.to_numeric(enriched_df["num_comments"], errors="coerce").fillna(0).astype(np.int32)
        enriched_df["created_utc"] = pd.to_numeric(enriched_df["created_utc"], errors="coerce").fillna(0).astype(np.int64)

        enriched_tbl = pa.Table.from_pandas(enriched_df, schema=ENRICH_SCHEMA, preserve_index=False)
        out_enriched_path = out_enriched_root / day / "submissions_with_tickers.parquet"
        write_parquet(enriched_tbl, out_enriched_path, schema=ENRICH_SCHEMA)
        print(f"  - wrote enriched: {out_enriched_path} ({enriched_tbl.num_rows} rows)")

        # ----- Write exploded day (may be 0) -----
        if exploded_rows:
            exp_tbl = pa.Table.from_pylist(exploded_rows, schema=EXPLODED_SCHEMA)
            out_exploded_path = out_exploded_root / day / "submission_tickers.parquet"
            write_parquet(exp_tbl, out_exploded_path, schema=EXPLODED_SCHEMA)
            print(f"  - wrote exploded: {out_exploded_path} ({exp_tbl.num_rows} rows)")
        else:
            print("  - no exploded rows (no tickers found)")

if __name__ == "__main__":
    main()