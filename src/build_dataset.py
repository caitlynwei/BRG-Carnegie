"""
Dataset Builder for Cultural Exchange Research Data
Reads from CSV files in data/raw/ and outputs to data/processed/
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).parent.parent  # assumes script is in src/
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Create output dir if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# COUNTRY MAPPING
# ---------------------------------------------------------
COUNTRY_TO_ISO = {
    # EU variants
    "EU": "EU",
    "EU-27": "EU",
    "EU27": "EU",
    "European Union": "EU",
    # US variants
    "United States": "USA",
    "USA": "USA",
    "US": "USA",
    # China
    "China (Including Hong Kong)": "CHN",
    "CHINA (Including Hong Kong)": "CHN",
    "China": "CHN",
    # Others
    "Japan": "JPN",
    "India": "IND",
    # Add more as needed
}

# ---------------------------------------------------------
# CONFIG: Define each source table
# ---------------------------------------------------------
# Each entry tells the script how to process one CSV file
TABLES = [
    {
        "file": "trade_services.csv",
        "id_col": "country",
        "year_start": 2015,
        "year_end": 2024,
        "sector": "Cultural Exchange",
        "metric": "International Trade in Services - Travel",
        "unit": "Euro",
        "multiplier": 1_000_000,
        "source": "Eurostat",
        "source_url": "https://ec.europa.eu/eurostat/databrowser",
    },
    {
        "file": "tourism_arrivals.csv",
        "id_col": "country",
        "year_start": 2016,
        "year_end": 2025,
        "sector": "Cultural Exchange",
        "metric": "International Tourism Arrivals",
        "unit": "Euro",
        "multiplier": 1_000_000,
        "source": "Eurostat",
        "source_url": "https://ec.europa.eu/eurostat/databrowser",
    },
    # Add more tables here as needed
]

# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------
def reshape_table(
    df: pd.DataFrame,
    id_col: str,
    year_start: int,
    year_end: int,
    sector: str,
    metric: str,
    unit: str,
    source: str = "",
    source_url: str = "",
    multiplier: float = 1.0
) -> pd.DataFrame:
    
    year_cols = [str(y) for y in range(year_start, year_end + 1)]
    year_cols = [c for c in year_cols if c in df.columns]
    
    long_df = df.melt(
        id_vars=[id_col],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    
    long_df = long_df.rename(columns={id_col: "country_name"})
    long_df["year"] = long_df["year"].astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    
    # multiplier to convert units
    long_df["value"] = long_df["value"] * multiplier
    
    long_df["iso"] = long_df["country_name"].map(COUNTRY_TO_ISO)
    
    # Flag unmapped countries
    unmapped = long_df[long_df["iso"].isna()]["country_name"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠️  Unmapped countries in {metric}: {unmapped.tolist()}")
    
    long_df["sector"] = sector
    long_df["metric"] = metric
    long_df["unit"] = unit
    long_df["source"] = source
    long_df["source_url"] = source_url
    
    return long_df[["iso", "year", "sector", "metric", "value", "unit", "source_url"]]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    all_tables = []
    
    for config in TABLES:
        filepath = RAW_DIR / config["file"]
        
        if not filepath.exists():
            print(f"Skipping {config['file']} — file not found")
            continue
        
        print(f"Reading {config['file']}...")
        df = pd.read_csv(filepath)
        
        long_df = reshape_table(
            df=df,
            id_col=config["id_col"],
            year_start=config["year_start"],
            year_end=config["year_end"],
            sector=config["sector"],
            metric=config["metric"],
            unit=config["unit"],
            source=config.get("source", ""),
            source_url=config.get("source_url", ""),
            multiplier=config.get("multiplier", 1.0),
        )
        all_tables.append(long_df)
    
    if not all_tables:
        print("No tables processed. Exiting.")
        return
    
    combined = pd.concat(all_tables, ignore_index=True)
    combined = combined.sort_values(["iso", "metric", "year"]).reset_index(drop=True)
    
    # Save
    output_path = PROCESSED_DIR / "cultural_exchange_dataset.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"\nDone.")
    print(f"   Rows: {len(combined)}")
    print(f"   Metrics: {combined['metric'].nunique()}")
    print(f"   Countries: {combined['iso'].nunique()}")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()