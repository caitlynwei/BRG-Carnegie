"""
Utility functions for filtering and visualizing the cultural exchange dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the processed dataset.
    
    Args:
        path: Path to CSV. If None, looks for default location.
    
    Returns:
        DataFrame with the cultural exchange data.
    """
    if path is None:
        # Try common locations
        candidates = [
            Path("data/processed/cultural_exchange_dataset.csv"),
            Path("../data/processed/cultural_exchange_dataset.csv"),
            Path(__file__).parent.parent / "data" / "processed" / "cultural_exchange_dataset.csv",
        ]
        for p in candidates:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError("Could not find dataset. Provide path explicitly.")
    
    return pd.read_csv(path)


# ---------------------------------------------------------
# FILTERING FUNCTIONS
# ---------------------------------------------------------
def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter dataset to a single year."""
    return df[df["year"] == year].reset_index(drop=True)


def filter_by_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter dataset to a range of years (inclusive)."""
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].reset_index(drop=True)


def filter_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Filter dataset to a single metric."""
    return df[df["metric"] == metric].reset_index(drop=True)


def filter_by_country(df: pd.DataFrame, iso: str) -> pd.DataFrame:
    """Filter dataset to a single country by ISO code."""
    return df[df["iso"] == iso].reset_index(drop=True)


def filter_by_countries(df: pd.DataFrame, iso_list: list) -> pd.DataFrame:
    """Filter dataset to multiple countries by ISO codes."""
    return df[df["iso"].isin(iso_list)].reset_index(drop=True)


def filter_by_sector(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Filter dataset to a single sector."""
    return df[df["sector"] == sector].reset_index(drop=True)


# ---------------------------------------------------------
# SUMMARY FUNCTIONS
# ---------------------------------------------------------
def list_metrics(df: pd.DataFrame) -> list:
    """Return list of unique metrics in the dataset."""
    return df["metric"].unique().tolist()


def list_countries(df: pd.DataFrame) -> list:
    """Return list of unique ISO codes in the dataset."""
    return df["iso"].dropna().unique().tolist()


def list_years(df: pd.DataFrame) -> list:
    """Return sorted list of years in the dataset."""
    return sorted(df["year"].unique().tolist())


def summarize(df: pd.DataFrame) -> dict:
    """Return a summary of the dataset contents."""
    return {
        "rows": len(df),
        "metrics": list_metrics(df),
        "countries": list_countries(df),
        "years": list_years(df),
        "year_range": (df["year"].min(), df["year"].max()),
    }


# ---------------------------------------------------------
# PIVOT FUNCTIONS (for easier analysis)
# ---------------------------------------------------------
def pivot_by_year(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pivot a metric to have countries as rows and years as columns.
    
    Returns:
        DataFrame with ISO as index, years as columns, values as cells.
    """
    subset = filter_by_metric(df, metric)
    return subset.pivot(index="iso", columns="year", values="value")


def pivot_by_country(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pivot a metric to have years as rows and countries as columns.
    
    Returns:
        DataFrame with year as index, ISO codes as columns, values as cells.
    """
    subset = filter_by_metric(df, metric)
    return subset.pivot(index="year", columns="iso", values="value")


# ---------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_metric_over_time(
    df: pd.DataFrame,
    metric: str,
    countries: Optional[list] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a metric over time for selected countries.
    
    Args:
        df: The dataset
        metric: Which metric to plot
        countries: List of ISO codes. If None, plots all countries.
        title: Plot title. If None, uses metric name.
        ylabel: Y-axis label. If None, uses unit from data.
        figsize: Figure size tuple
        save_path: If provided, saves figure to this path
    
    Returns:
        matplotlib Figure object
    """
    subset = filter_by_metric(df, metric)
    
    if countries is not None:
        subset = filter_by_countries(subset, countries)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for iso in subset["iso"].unique():
        country_data = subset[subset["iso"] == iso].sort_values("year")
        ax.plot(country_data["year"], country_data["value"], marker="o", label=iso)
    
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel or subset["unit"].iloc[0] if len(subset) > 0 else "Value")
    ax.set_title(title or metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show integer years
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    year: int,
    countries: Optional[list] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing countries for a single metric and year.
    
    Args:
        df: The dataset
        metric: Which metric to plot
        year: Which year to compare
        countries: List of ISO codes. If None, plots all countries.
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size tuple
        save_path: If provided, saves figure to this path
    
    Returns:
        matplotlib Figure object
    """
    subset = filter_by_metric(df, metric)
    subset = filter_by_year(subset, year)
    
    if countries is not None:
        subset = filter_by_countries(subset, countries)
    
    subset = subset.sort_values("value", ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(subset["iso"], subset["value"])
    
    ax.set_xlabel(ylabel or subset["unit"].iloc[0] if len(subset) > 0 else "Value")
    ax.set_ylabel("Country")
    ax.set_title(title or f"{metric} ({year})")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_all_metrics(
    df: pd.DataFrame,
    countries: Optional[list] = None,
    figsize: tuple = (12, 5),
    save_dir: Optional[str] = None
) -> list:
    """
    Generate a time series plot for each metric in the dataset.
    
    Args:
        df: The dataset
        countries: List of ISO codes to include. If None, includes all.
        figsize: Figure size for each plot
        save_dir: If provided, saves all figures to this directory
    
    Returns:
        List of matplotlib Figure objects
    """
    metrics = list_metrics(df)
    figs = []
    
    for metric in metrics:
        save_path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            safe_name = metric.lower().replace(" ", "_").replace("-", "_")
            save_path = f"{save_dir}/{safe_name}.png"
        
        fig = plot_metric_over_time(
            df, metric, countries=countries, figsize=figsize, save_path=save_path
        )
        figs.append(fig)
    
    return figs


# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load data
    df = load_dataset()
    
    # Print summary
    print("=== Dataset Summary ===")
    summary = summarize(df)
    for key, val in summary.items():
        print(f"  {key}: {val}")
    
    # Example: filter by metric
    print("\n=== Tourism Arrivals Data ===")
    tourism = filter_by_metric(df, "International Tourism Arrivals")
    print(tourism.head(10))
    
    # Example: pivot table
    print("\n=== Tourism Arrivals by Year (Pivot) ===")
    pivot = pivot_by_year(df, "International Tourism Arrivals")
    print(pivot)
    
    # Example: plot
    print("\n=== Generating Plots ===")
    plot_metric_over_time(
        df,
        metric="International Tourism Arrivals",
        countries=["USA", "EU", "CHN"],
        save_path="tourism_plot.png"
    )
    
    plot_metric_comparison(
        df,
        metric="International Tourism Arrivals",
        year=2019,
        save_path="tourism_2019_comparison.png"
    )
    
    plt.show()