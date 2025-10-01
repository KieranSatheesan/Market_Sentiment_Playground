# scripts/build_universe.py
import argparse, os, re, unicodedata
import pandas as pd

# --- helpers ---
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")  # allow dots/dashes (e.g., BRK.B, RDS-A)

LEGAL_STOPWORDS = {
    "inc", "inc.", "corp", "corp.", "co", "co.", "ltd", "ltd.", "plc", "nv", "n.v.",
    "sa", "s.a.", "ag", "se", "ab", "oyj", "spa", "group", "holdings", "holding",
    "company", "the", "class", "series", "adr", "ads", "pte", "llc", "lp", "l.p.",
    "common", "stock", "stocks", "share", "shares", "corporation", "technologies",
    "platforms", "systems", "solutions", "international", "global", "bancorp",
    "financial", "capital", "trust", "properties", "energy", "pharmaceuticals"
}

def norm_ascii_lower(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())

def basic_aliases(name: str) -> list[str]:
    """
    Make alias variants from a company name including single words.
    """
    base = norm_ascii_lower(name)
    if not base:
        return []

    # remove punctuation commonly found in names
    cleaned = re.sub(r"[&.,()\-/]", " ", base)
    tokens = [t for t in cleaned.split() if t and t not in LEGAL_STOPWORDS]

    aliases = set()
    if tokens:
        # Add single word aliases (NEW - this fixes the main issue)
        for token in tokens:
            if len(token) >= 3:  # avoid short words
                aliases.add(token)
        
        # Existing multi-word aliases
        aliases.add(" ".join(tokens))  # full tokens w/o legal words
        if len(tokens) >= 2:
            aliases.add(" ".join(tokens[:2]))
        if len(tokens) >= 3:
            aliases.add(" ".join(tokens[:3]))

    return sorted(a for a in aliases if len(a) >= 3)

def load_symbols(path: str, exchange: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # try to find symbol/name columns
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("code")
    name_col = cols.get("name") or cols.get("company") or cols.get("company_name")

    if sym_col is None or name_col is None:
        raise ValueError(f"Could not find Symbol/Name columns in {path}. Got columns={list(df.columns)}")

    out = df[[sym_col, name_col]].rename(columns={sym_col: "ticker", name_col: "company_name"}).copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["company_name"] = out["company_name"].astype(str).str.strip()
    out["exchange"] = exchange

    # filter to plausible tickers
    out = out[out["ticker"].apply(lambda t: bool(TICKER_RE.match(t)))]
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    return out

def read_manual_aliases(path: str | None) -> pd.DataFrame:
    """
    Optional: CSV with columns: alias,ticker
    All lowercased; we'll normalize to ascii lowercase here anyway.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["alias","ticker"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    alias_col = cols.get("alias")
    ticker_col = cols.get("ticker")
    if alias_col is None or ticker_col is None:
        raise ValueError(f"manual aliases file needs alias,ticker columns. got {list(df.columns)}")
    out = df[[alias_col, ticker_col]].rename(columns={alias_col: "alias", ticker_col: "ticker"}).copy()
    out["alias"] = out["alias"].astype(str).map(norm_ascii_lower)
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out.dropna().drop_duplicates()
    return out

def main():
    ap = argparse.ArgumentParser(description="Build unified ticker universe CSV from exchange symbol lists.")
    ap.add_argument("--nyse", required=True, help="Path to NYSE symbols CSV")
    ap.add_argument("--nasdaq", required=True, help="Path to NASDAQ symbols CSV")
    ap.add_argument("--manual", default=None, help="(Optional) manual alias CSV with columns alias,ticker")
    ap.add_argument("--out", default="ref/ticker_universe.csv", help="Output CSV path")
    args = ap.parse_args()

    # Validate input files exist
    if not os.path.exists(args.nyse):
        raise FileNotFoundError(f"NYSE file not found: {args.nyse}")
    if not os.path.exists(args.nasdaq):
        raise FileNotFoundError(f"NASDAQ file not found: {args.nasdaq}")

    print("Loading symbol data...")
    nyse = load_symbols(args.nyse, exchange="NYSE")
    nasdaq = load_symbols(args.nasdaq, exchange="NASDAQ")

    print(f"NYSE symbols: {len(nyse)}, NASDAQ symbols: {len(nasdaq)}")
    
    uni = pd.concat([nyse, nasdaq], ignore_index=True)
    # If same ticker appears in both, keep first occurrence (order: NYSE first, then NASDAQ as loaded)
    uni = uni.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    print("Generating aliases...")
    # generate light alias candidates from company_name
    uni["aliases_auto"] = uni["company_name"].apply(lambda s: ";".join(basic_aliases(s)))

    # merge in manual aliases (if provided)
    man = read_manual_aliases(args.manual)
    if not man.empty:
        print(f"Loaded {len(man)} manual aliases")
        # aggregate manual aliases per ticker into semicolon list
        man_agg = man.groupby("ticker")["alias"].apply(lambda s: ";".join(sorted(set(s)))).reset_index(name="aliases_manual")
        uni = uni.merge(man_agg, on="ticker", how="left")
    else:
        uni["aliases_manual"] = ""

    # combine alias sources → one lowercase ascii-normalized field
    def combine_aliases(row):
        parts = []
        if isinstance(row.get("aliases_manual"), str) and row["aliases_manual"]:
            parts.extend([a.strip() for a in row["aliases_manual"].split(";") if a.strip()])
        if isinstance(row.get("aliases_auto"), str) and row["aliases_auto"]:
            parts.extend([a.strip() for a in row["aliases_auto"].split(";") if a.strip()])
        # include the raw company_name (normalized) too
        cname = norm_ascii_lower(row.get("company_name", ""))
        if cname:
            parts.append(cname)
        # de-dupe and remove trivially short tokens
        uniq = sorted(set(p for p in parts if len(p) >= 3))
        return ";".join(uniq)

    uni["aliases"] = uni.apply(combine_aliases, axis=1)

    # final tidy columns
    out_cols = ["ticker", "company_name", "exchange", "aliases"]
    uni = uni[out_cols].sort_values("ticker").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    uni.to_csv(args.out, index=False)

    # quick stats
    n_rows = len(uni)
    n_with_aliases = int((uni["aliases"].str.len() > 0).sum())
    
    # Show some examples
    print("\nSample aliases generated:")
    sample_tickers = ['GME', 'AAPL', 'MSFT', 'META', 'NFLX']
    for ticker in sample_tickers:
        if ticker in uni['ticker'].values:
            aliases = uni[uni['ticker'] == ticker]['aliases'].iloc[0]
            print(f"  {ticker}: {aliases}")
    
    print(f"\n✅ Wrote {args.out}")
    print(f"   Total rows: {n_rows}, With aliases: {n_with_aliases}")

if __name__ == "__main__":
    main()