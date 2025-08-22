# viz_suite.py
# Professional visualization suite for SHM Heavy Equipment Analysis
# Fixes overplotting, mixed scales, inconsistent styling, and adds high-impact visualizations

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import professional theme system
try:
    from .viz_theme import (
        set_viz_theme, k_formatter, currency_formatter, percentage_formatter,
        shade_splits, log_price, add_value_labels, create_confidence_ribbon,
        apply_log_scale_safely, set_currency_axis, create_subplot_grid,
        get_color_palette, COLORS
    )
except ImportError:
    from viz_theme import (
        set_viz_theme, k_formatter, currency_formatter, percentage_formatter,
        shade_splits, log_price, add_value_labels, create_confidence_ribbon,
        apply_log_scale_safely, set_currency_axis, create_subplot_grid,
        get_color_palette, COLORS
    )

# Reuse existing data loading utilities from shm_explore.py
try:
    from shm_explore import normalize_missing, to_snake
    CSV_PATH = "data/raw/Bit_SHM_data.csv"  # Updated path for new directory structure
except ImportError:
    # Fallback implementations
    def to_snake(s: str) -> str:
        import re
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', str(s).strip())
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()
    
    def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
        LOWER_MISSING_STRINGS = {"none or unspecified", "unknown", "n/a", "na", ""}
        for col in df.columns:
            if df[col].dtype == object:
                ser = df[col].astype(str)
                df[col] = ser.where(~ser.str.strip().str.lower().isin(LOWER_MISSING_STRINGS), np.nan)
        return df
    
    CSV_PATH = "data/raw/Bit_SHM_data.csv"

def _ensure_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric 'price' column exists by coalescing common alternatives.

    This makes plotting functions resilient when called with raw dataframes
    that use 'sales_price' or 'saleprice' instead of the canonical 'price'.
    """
    if 'price' not in df.columns:
        for cand in ("sales_price", "saleprice", "price"):
            if cand in df.columns and cand != 'price':
                out = df.copy()
                out['price'] = pd.to_numeric(out[cand], errors='coerce')
                return out
    return df

def load_clean_df(path: str | Path = CSV_PATH) -> pd.DataFrame:
    """
    Load and clean data with consistent column naming and feature engineering.
    Reuses existing cleaning logic for consistency.
    """
    # Try multiple possible locations
    path_candidates = [
        Path(path),
        Path("data/raw/Bit_SHM_data.csv"),
        Path("data/Bit_SHM_data.csv"),
        Path("Bit_SHM_data.csv"),
        Path("../data/raw/Bit_SHM_data.csv"),
        Path("../data/Bit_SHM_data.csv")
    ]
    
    df = None
    for candidate in path_candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, low_memory=False)
            break
    
    if df is None:
        raise FileNotFoundError(f"Could not find data file in any of: {path_candidates}")
    
    # Standardize column names and coalesce aliases
    df.columns = [to_snake(c) for c in df.columns]
    aliases = {
        "saledate": "sales_date",
        "sale_date": "sales_date",
        "date": "sales_date",
        "saleprice": "sales_price",
        "price": "sales_price",
        "target": "sales_price",
        "machinehourscurrentmeter": "machinehours_currentmeter",
        "machine_hours": "machinehours_currentmeter",
        "machine_hours_current_meter": "machinehours_currentmeter",
        "hours": "machinehours_currentmeter",
        "yearmade": "year_made",
        "year": "year_made",
        "salesid": "sales_id",
        "machineid": "machine_id",
    }
    rename_map = {c: aliases[c] for c in df.columns if c in aliases}
    if rename_map:
        df = df.rename(columns=rename_map)
    df = normalize_missing(df)
    
    # Parse dates
    date_candidates = [c for c in df.columns if "sale" in c and "date" in c]
    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["sale_year"] = df[date_col].dt.year
        df["sale_month"] = df[date_col].dt.month
        df["sale_quarter"] = df[date_col].dt.quarter
    
    # Standardize target price column
    price_candidates = ["sales_price", "saleprice", "price"]
    for candidate in price_candidates:
        if candidate in df.columns:
            df["price"] = pd.to_numeric(df[candidate], errors="coerce")
            break
    
    # Feature engineering - Age
    if "year_made" in df.columns and "sale_year" in df.columns:
        df["age_years"] = df["sale_year"] - df["year_made"]
        # Clean implausible ages
        df.loc[df["age_years"] < 0, "age_years"] = np.nan
        df.loc[df["age_years"] > 50, "age_years"] = np.nan  # Equipment >50 years old is likely data error
        df.loc[df["year_made"] < 1950, "year_made"] = np.nan  # Clean implausible years
        df.loc[df["year_made"] == 1000, "year_made"] = np.nan  # Explicitly handle 1000 as missing
    
    # Feature engineering - Usage
    if "machinehours_currentmeter" in df.columns:
        # Treat zeros as missing
        df.loc[df["machinehours_currentmeter"].fillna(0) == 0, "machinehours_currentmeter"] = np.nan
        df["log1p_hours"] = np.log1p(df["machinehours_currentmeter"].fillna(0))
        
        # Calculate usage intensity
        if "age_years" in df.columns:
            df["hours_per_year"] = df["machinehours_currentmeter"] / df["age_years"].clip(lower=0.5)
    
    # Usage band harmonization
    if "usage_band" in df.columns:
        df["usage_band"] = df["usage_band"].astype("category")
    
    print(f"Loaded and cleaned {len(df):,} records with {len(df.columns)} features")
    return df

# ---------- FIXED/IMPROVED VERSIONS OF EXISTING VISUALS ----------

def price_distribution_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Fixed price distribution analysis.
    - Uses log scale for better readability
    - QQ plot on log-transformed target (not raw)
    - Professional styling
    """
    set_viz_theme()
    df = _ensure_price_column(df)
    if 'price' not in df.columns:
        return None
    fig, axes = create_subplot_grid(1, 2, figsize=(14, 6))
    axes = axes.flatten()
    
    # Raw price with log x-axis for readability
    price_data = df["price"].dropna()
    sns.histplot(price_data, bins=60, ax=axes[0], color=COLORS['primary'], alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_title("Price Distribution (Log Scale)")
    axes[0].set_xlabel("Price ($, log scale)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(price_data.mean(), color=COLORS['danger'], linestyle='--', 
                   label=f'Mean: ${price_data.mean():,.0f}')
    axes[0].axvline(price_data.median(), color=COLORS['warning'], linestyle='--',
                   label=f'Median: ${price_data.median():,.0f}')
    axes[0].legend()
    
    # Log-transformed price distribution with KDE
    log_price_data = np.log1p(price_data)
    sns.histplot(log_price_data, kde=True, bins=60, ax=axes[1], 
                color=COLORS['secondary'], alpha=0.7)
    axes[1].set_title("log1p(Price) Distribution + KDE")
    axes[1].set_xlabel("log1p(Price)")
    axes[1].set_ylabel("Density")
    
    fig.suptitle("Heavy Equipment: Target Variable Diagnostics", y=0.98)
    return fig

def age_vs_price_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Fixed age vs price analysis.
    - Uses log(price) to handle heteroskedasticity
    - 2D density plot to avoid overplotting
    - Robust trend with LOWESS
    - Depreciation curve with percentile bands
    """
    set_viz_theme()
    df = _ensure_price_column(df)
    # Derive age_years if missing but ingredients are available
    if 'age_years' not in df.columns:
        sale_year = None
        if 'sale_year' in df.columns:
            sale_year = df['sale_year']
        elif 'sales_date' in df.columns:
            sale_year = pd.to_datetime(df['sales_date'], errors='coerce').dt.year
        if sale_year is not None and 'year_made' in df.columns:
            tmp = df.copy()
            tmp['age_years'] = sale_year - tmp['year_made']
            df = tmp
        else:
            return None
    if 'price' not in df.columns:
        return None
    data = df.dropna(subset=["age_years", "price"]).copy()
    data["log_price"] = np.log1p(data["price"])
    
    fig, axes = create_subplot_grid(1, 2, figsize=(15, 6))
    axes = axes.flatten()
    
    # 2D density plot to avoid overplotting
    sns.kdeplot(
        data=data, x="age_years", y="log_price",
        fill=True, thresh=0.05, levels=15, cmap="Blues",
        ax=axes[0]
    )
    
    # Add robust trend line (LOWESS)
    sample_data = data.sample(min(5000, len(data)))  # Sample for performance
    sns.regplot(
        data=sample_data, x="age_years", y="log_price",
        scatter_kws={"s": 10, "alpha": 0.2}, 
        lowess=True, 
        line_kws={"color": COLORS['danger'], "linewidth": 2},
        ax=axes[0]
    )
    
    axes[0].set_title("Age vs log1p(Price): Density + Robust Trend")
    axes[0].set_xlabel("Equipment Age (years)")
    axes[0].set_ylabel("log1p(Price)")
    
    # Depreciation curve with robust central tendency + percentile ribbon
    # Limit to reasonable age range
    age_bins = pd.cut(data["age_years"], bins=np.arange(0, min(data["age_years"].max(), 40)+2, 1))
    depreciation_stats = (data.groupby(age_bins)["price"]
                         .agg(median="median", 
                              q10=lambda x: x.quantile(0.10),
                              q90=lambda x: x.quantile(0.90),
                              count="count")
                         .query("count >= 10")  # Only reliable bins
                         .dropna())
    
    if len(depreciation_stats) > 0:
        x_positions = range(len(depreciation_stats))
        axes[1].plot(x_positions, depreciation_stats["median"], 
                    'o-', color=COLORS['primary'], linewidth=2, markersize=6,
                    label='Median Price')
        
        # Add confidence ribbon
        axes[1].fill_between(x_positions, 
                           depreciation_stats["q10"], 
                           depreciation_stats["q90"], 
                           alpha=0.2, color=COLORS['primary'],
                           label='10th-90th percentile')
        
        axes[1].set_title("Equipment Depreciation Curve")
        axes[1].set_xlabel("Age Bins (years)")
        axes[1].set_ylabel("Price ($)")
        axes[1].set_xticks(x_positions[::2])  # Every other label
        axes[1].set_xticklabels([str(depreciation_stats.index[i]).split(',')[0].replace('(', '') 
                               for i in x_positions[::2]], rotation=45)
        set_currency_axis(axes[1], 'y')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Equipment Age vs Price Analysis (Fixed & Robust)", y=0.98)
    return fig

def product_group_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Fixed product group analysis.
    - Replaces pie charts with horizontal bars
    - Adds confidence intervals
    - Shows price distributions
    """
    set_viz_theme()
    df = _ensure_price_column(df)
    
    if "product_group" not in df.columns:
        return None
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
    
    # Volume by product group (horizontal bar)
    ax1 = fig.add_subplot(gs[0, 0])
    volume_data = (df["product_group"].value_counts()
                  .head(8)  # Top 8 for readability
                  .sort_values(ascending=True))  # Ascending for better visual flow
    
    bars = ax1.barh(range(len(volume_data)), volume_data.values, 
                    color=get_color_palette(len(volume_data)))
    ax1.set_yticks(range(len(volume_data)))
    ax1.set_yticklabels(volume_data.index)
    ax1.set_title("Sales Volume by Product Group")
    ax1.set_xlabel("Number of Sales")
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, volume_data.values)):
        ax1.text(bar.get_width() + max(volume_data) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:,}', ha='left', va='center', fontsize=9)
    
    # Median price by group with confidence intervals
    ax2 = fig.add_subplot(gs[0, 1])
    price_data = df.dropna(subset=["price", "product_group"])
    
    # Calculate median and confidence intervals
    group_stats = []
    for group in volume_data.index:
        group_prices = price_data[price_data["product_group"] == group]["price"]
        if len(group_prices) >= 10:  # Minimum for reliable stats
            median_price = group_prices.median()
            q25 = group_prices.quantile(0.25)
            q75 = group_prices.quantile(0.75)
            group_stats.append({
                'group': group,
                'median': median_price,
                'q25': q25,
                'q75': q75
            })
    
    if group_stats:
        group_df = pd.DataFrame(group_stats).sort_values('median')
        
        ax2.errorbar(group_df['median'], range(len(group_df)),
                    xerr=[group_df['median'] - group_df['q25'],
                          group_df['q75'] - group_df['median']],
                    fmt='o', capsize=5, capthick=2,
                    color=COLORS['primary'], markersize=8)
        
        ax2.set_yticks(range(len(group_df)))
        ax2.set_yticklabels(group_df['group'])
        ax2.set_xscale("log")
        ax2.set_title("Median Price by Group (25th-75th percentile)")
        ax2.set_xlabel("Price ($, log scale)")
        ax2.grid(True, alpha=0.3)
    
    # Price distributions by product group
    ax3 = fig.add_subplot(gs[1, :])
    
    top_groups = volume_data.head(6).index  # Top 6 for violin plot readability
    violin_data = price_data[price_data["product_group"].isin(top_groups)].copy()
    violin_data["log_price"] = np.log1p(violin_data["price"])
    
    if not violin_data.empty:
        sns.violinplot(data=violin_data, x="product_group", y="log_price",
                      inner="quartile", ax=ax3, palette="Set2")
        ax3.set_title("Price Distribution by Product Group (log scale)")
        ax3.set_xlabel("Product Group")
        ax3.set_ylabel("log1p(Price)")
        ax3.tick_params(axis='x', rotation=45)
    
    fig.suptitle("Product Group Analysis (Professional Format)", y=0.98)
    return fig

def temporal_trends_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Fixed temporal analysis.
    - Separate aligned panels instead of dual axes
    - Shade train/validation/test periods
    - Professional styling
    """
    set_viz_theme()
    
    if "sale_year" not in df.columns:
        return None
    
    temporal_data = df.dropna(subset=["sale_year", "price"])
    yearly_stats = (temporal_data.groupby("sale_year")["price"]
                   .agg(count="count", median="median", mean="mean", std="std")
                   .reset_index())
    
    fig, axes = create_subplot_grid(2, 1, figsize=(14, 10))
    axes = axes.flatten()
    
    # Sales volume over time
    bars = axes[0].bar(yearly_stats["sale_year"], yearly_stats["count"], 
                      alpha=0.7, color=COLORS['primary'])
    axes[0].set_ylabel("Sales Volume")
    axes[0].set_title("Sales Volume Over Time")
    axes[0].grid(True, alpha=0.3)
    shade_splits(axes[0])
    
    # Median price over time with error bands
    axes[1].plot(yearly_stats["sale_year"], yearly_stats["median"], 
                'o-', linewidth=2, markersize=6, color=COLORS['secondary'],
                label='Median Price')
    
    # Add standard deviation bands
    axes[1].fill_between(yearly_stats["sale_year"],
                        yearly_stats["median"] - yearly_stats["std"]/2,
                        yearly_stats["median"] + yearly_stats["std"]/2,
                        alpha=0.2, color=COLORS['secondary'],
                        label='+/-0.5 Std Dev')
    
    axes[1].set_ylabel("Price ($)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Median Price Trends Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    set_currency_axis(axes[1], 'y')
    shade_splits(axes[1])
    
    # Share x-axis for better alignment
    axes[0].sharex(axes[1])
    
    fig.suptitle("Temporal Market Analysis (Aligned Panels)", y=0.98)
    return fig

def usage_vs_price_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Fixed usage vs price analysis.
    - 2D histogram to show density patterns
    - Proper log scaling
    - Clear annotations
    """
    set_viz_theme()
    
    if "machinehours_currentmeter" not in df.columns:
        return None
    
    usage_data = df.dropna(subset=["machinehours_currentmeter", "price"]).copy()
    usage_data["log_price"] = np.log1p(usage_data["price"])
    
    # Create 2D histogram figure
    fig = plt.figure(figsize=(12, 8))
    
    # Main 2D histogram
    ax = fig.add_subplot(111)
    
    # Create 2D histogram
    h = ax.hist2d(usage_data["machinehours_currentmeter"], usage_data["log_price"],
                 bins=50, cmap='Blues', alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Number of Sales', rotation=270, labelpad=20)
    
    ax.set_xscale("log")
    ax.set_xlabel("Machine Hours (log scale)")
    ax.set_ylabel("log1p(Price)")
    ax.set_title("Machine Hours vs Price Density Map")
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    sample_data = usage_data.sample(min(5000, len(usage_data)))
    z = np.polyfit(np.log1p(sample_data["machinehours_currentmeter"]), 
                   sample_data["log_price"], 1)
    p = np.poly1d(z)
    
    x_trend = np.logspace(np.log10(usage_data["machinehours_currentmeter"].min()),
                         np.log10(usage_data["machinehours_currentmeter"].max()), 100)
    y_trend = p(np.log1p(x_trend))
    
    ax.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.8,
           label=f'Trend: slope = {z[0]:.3f}')
    ax.legend()
    
    return fig

# ---------- FIVE ADDITIONAL HIGH-IMPACT VISUALIZATIONS ----------

def missingness_overview_fig(df: pd.DataFrame, top_n: int = 25) -> plt.Figure:
    """
    Data quality radar showing missingness patterns.
    - Top-N missingness bar chart
    - Missingness by year heatmap for key features
    """
    set_viz_theme()
    
    # Calculate missingness percentages
    missingness = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    
    fig, axes = create_subplot_grid(1, 2, figsize=(16, 8))
    axes = axes.flatten()
    
    # Missingness bar chart
    bars = axes[0].barh(range(len(missingness)), missingness.values, 
                       color=get_color_palette(len(missingness)))
    axes[0].set_yticks(range(len(missingness)))
    axes[0].set_yticklabels(missingness.index)
    axes[0].set_title(f"Top {top_n} Features by Missingness")
    axes[0].set_xlabel("Missing Data (%)")
    axes[0].grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars, missingness.values)):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=9)
    
    # Missingness by year heatmap (if temporal data available)
    if "sale_year" in df.columns:
        # Focus on engineered features and critical fields only
        key_features = ["age_years", "hours_per_year", "machinehours_currentmeter", 
                       "year_made", "product_group"]
        available_features = [f for f in key_features if f in df.columns]
        
        if available_features:
            yearly_missing = (df.groupby("sale_year")[available_features]
                            .apply(lambda x: x.isna().mean() * 100))
            
            # Only annotate if not too many cells (avoid clutter)
            annot = len(yearly_missing) * len(available_features) <= 50
            
            sns.heatmap(yearly_missing.T, cmap="Reds", cbar_kws={"label": "% Missing"},
                       ax=axes[1], annot=annot, fmt='.0f', 
                       linewidths=0.5, linecolor='gray')
            axes[1].set_title("Missingness by Year (Engineered & Key Features)")
            axes[1].set_xlabel("Year")
            axes[1].set_ylabel("Feature")
            # Rotate x labels for better readability
            axes[1].tick_params(axis='x', rotation=45)
    
    fig.suptitle("Data Quality Assessment", y=0.98)
    return fig

def auctioneer_effect_fig(df: pd.DataFrame, top_k: int = 12) -> plt.Figure:
    """
    Auctioneer pricing effect analysis.
    - Median price vs volume scatter
    - Highlights suspicious ID=1000
    """
    set_viz_theme()
    
    if "auctioneer_id" not in df.columns or "price" not in df.columns:
        return None
    
    auctioneer_data = df.dropna(subset=["auctioneer_id", "price"]).copy()
    
    # Get top auctioneers by volume
    top_auctioneers = auctioneer_data["auctioneer_id"].value_counts().head(top_k)
    filtered_data = auctioneer_data[auctioneer_data["auctioneer_id"].isin(top_auctioneers.index)]
    
    # Calculate statistics by auctioneer
    auctioneer_stats = (filtered_data.groupby("auctioneer_id")["price"]
                       .agg(median="median", count="count", mean="mean")
                       .sort_values("median"))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(auctioneer_stats["median"], auctioneer_stats.index,
                        s=auctioneer_stats["count"]/10,  # Size by volume
                        c=auctioneer_stats["mean"],  # Color by mean price
                        cmap="viridis", alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Price ($)', rotation=270, labelpad=20)
    
    ax.set_xscale("log")
    ax.set_xlabel("Median Price ($, log scale)")
    ax.set_ylabel("Auctioneer ID")
    ax.set_title("Auctioneer Pricing Effect: Median Price vs Volume")
    ax.grid(True, alpha=0.3)
    
    # Highlight ID=1000 if present
    if 1000 in auctioneer_stats.index:
        idx_1000 = list(auctioneer_stats.index).index(1000)
        ax.scatter(auctioneer_stats.loc[1000, "median"], idx_1000,
                  s=200, c='red', marker='x', linewidth=3,
                  label='ID=1000 (Suspicious)', zorder=10)
        ax.legend()
    
    # Add size legend
    legend_sizes = [100, 1000, 5000]
    legend_handles = [plt.scatter([], [], s=size/10, c='gray', alpha=0.7, 
                                 label=f'{size} sales') for size in legend_sizes]
    size_legend = ax.legend(handles=legend_handles, loc='upper left', 
                           title='Volume', frameon=True)
    ax.add_artist(size_legend)
    
    return fig

def age_hours_joint_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Utilization regime map showing age vs hours per year patterns.
    - Reveals different usage patterns (rental vs ownership)
    """
    set_viz_theme()
    
    required_cols = {"age_years", "hours_per_year"}
    if not required_cols.issubset(df.columns):
        return None
    
    joint_data = df.dropna(subset=list(required_cols)).copy()
    
    # Filter extreme outliers for better visualization
    joint_data = joint_data[
        (joint_data["hours_per_year"] > 0) & 
        (joint_data["hours_per_year"] < joint_data["hours_per_year"].quantile(0.95)) &
        (joint_data["age_years"] < 30)  # Focus on reasonable ages
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create 2D KDE plot
    sns.kdeplot(data=joint_data, x="age_years", y="hours_per_year",
               fill=True, cmap="viridis", levels=20, thresh=0.05, ax=ax)
    
    # Add scatter plot overlay (sample)
    sample_data = joint_data.sample(min(2000, len(joint_data)))
    ax.scatter(sample_data["age_years"], sample_data["hours_per_year"],
              alpha=0.3, s=10, c='white', edgecolors='black', linewidth=0.1)
    
    ax.set_xlabel("Equipment Age (years)")
    ax.set_ylabel("Hours per Year")
    ax.set_title("Equipment Utilization Regime Map: Age x Hours/Year")
    ax.grid(True, alpha=0.3)
    
    # Add regime annotations
    ax.text(2, 3000, "Heavy Use\n(Rental/Construction)", 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
           fontsize=10, ha='left')
    
    ax.text(15, 200, "Light Use\n(Occasional/Hobby)", 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
           fontsize=10, ha='center')
    
    return fig

def state_premia_fig(df: pd.DataFrame, min_sales: int = 500) -> plt.Figure:
    """
    Geographical pricing analysis by state.
    - Median price with confidence intervals
    - Only states with sufficient data
    """
    set_viz_theme()
    
    if "state_of_usage" not in df.columns or "price" not in df.columns:
        return None
    
    state_data = df.dropna(subset=["state_of_usage", "price"])
    
    # Calculate state statistics
    state_stats = (state_data.groupby("state_of_usage")["price"]
                  .agg(count="count", median="median", 
                       q25=lambda x: x.quantile(0.25),
                       q75=lambda x: x.quantile(0.75))
                  .query(f"count >= {min_sales}")
                  .sort_values("median"))
    
    if len(state_stats) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(state_stats) * 0.4)))
    
    # Create horizontal error bar plot
    y_positions = range(len(state_stats))
    ax.errorbar(state_stats["median"], y_positions,
               xerr=[state_stats["median"] - state_stats["q25"],
                     state_stats["q75"] - state_stats["median"]],
               fmt='o', capsize=5, capthick=2, markersize=8,
               color=COLORS['primary'])
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(state_stats.index)
    ax.set_xscale("log")
    ax.set_xlabel("Median Price ($, log scale)")
    ax.set_ylabel("State")
    ax.set_title(f"Geographic Price Premiums (States with >={min_sales} sales)")
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (state, stats) in enumerate(state_stats.iterrows()):
        ax.text(stats["median"] * 1.1, i, f'${stats["median"]:,.0f}\n({stats["count"]} sales)',
               va='center', fontsize=9)
    
    return fig

def temporal_heatmap_fig(df: pd.DataFrame) -> plt.Figure:
    """
    Calendar heatmap showing sales volume patterns.
    - Year x Month heatmap
    - Reveals seasonality at a glance
    """
    set_viz_theme()
    
    required_cols = {"sale_year", "sale_month"}
    if not required_cols.issubset(df.columns):
        return None
    
    # Create pivot table
    calendar_data = (df.groupby(["sale_year", "sale_month"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="sale_month", columns="sale_year", values="count")
                    .fillna(0))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(calendar_data, cmap="YlOrRd", cbar_kws={"label": "Sales Count"},
               ax=ax, annot=True, fmt='.0f', linewidth=0.5)
    
    ax.set_title("Sales Volume Calendar: Year x Month Heatmap")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    
    # Better month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels(month_labels, rotation=0)
    
    return fig

def save_all_figures(outdir: str = "plots") -> None:
    """
    Generate and save all visualization figures.
    Creates complete professional visualization suite.
    """
    output_dir = Path(outdir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data and generating visualizations...")
    df = load_clean_df()
    
    # Define all figures to generate
    figure_specs = {
        "01_price_distribution.png": ("Price Distribution Analysis", price_distribution_fig),
        "02_age_vs_price.png": ("Age vs Price Analysis", age_vs_price_fig),
        "03_product_groups.png": ("Product Group Analysis", product_group_fig),
        "04_temporal_trends.png": ("Temporal Trends Analysis", temporal_trends_fig),
        "05_usage_vs_price.png": ("Usage vs Price Analysis", usage_vs_price_fig),
        "06_missingness.png": ("Data Quality Assessment", missingness_overview_fig),
        "07_auctioneer_effect.png": ("Auctioneer Effect Analysis", auctioneer_effect_fig),
        "08_age_hours_joint.png": ("Utilization Regime Map", age_hours_joint_fig),
        "09_state_premia.png": ("Geographic Price Analysis", state_premia_fig),
        "10_temporal_heatmap.png": ("Calendar Heatmap", temporal_heatmap_fig),
    }
    
    # Generate each figure
    generated_count = 0
    for filename, (description, fig_func) in figure_specs.items():
        try:
            print(f"  Generating {description}...")
            fig = fig_func(df)
            
            if fig is not None:
                fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
                plt.close(fig)
                generated_count += 1
                print(f"    [OK] Saved: {filename}")
            else:
                print(f"    [SKIP] Skipped: {filename} (insufficient data)")
                
        except Exception as e:
            print(f"    [ERROR] Error generating {filename}: {e}")
            
    print(f"\nGenerated {generated_count}/{len(figure_specs)} visualizations in '{outdir}/'")
    
    # Generate summary report
    summary_path = output_dir / "visualization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SHM Heavy Equipment Analysis - Visualization Suite Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {generated_count}/{len(figure_specs)} figures\n")
        f.write(f"Data records analyzed: {len(df):,}\n")
        f.write(f"Features available: {len(df.columns)}\n\n")
        
        f.write("Figure Descriptions:\n")
        f.write("-" * 20 + "\n")
        for filename, (description, _) in figure_specs.items():
            status = "[OK]" if (output_dir / filename).exists() else "[MISSING]"
            f.write(f"{status} {filename}: {description}\n")
    
    print(f"Summary report saved: {summary_path}")

if __name__ == "__main__":
    save_all_figures(outdir="plots")
