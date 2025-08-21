# viz_theme.py
# Professional visualization theme system for SHM Heavy Equipment Analysis
# Ensures consistent visual language across all plots

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def set_viz_theme():
    """
    Set professional visualization theme with consistent styling.
    Single source of visual truth for all plots.
    """
    sns.set_theme(
        context="notebook", 
        style="whitegrid",
        rc={
            # Typography hierarchy
            "axes.titlesize": 14, 
            "axes.titleweight": "bold",
            "axes.labelsize": 12, 
            "figure.titlesize": 16,
            "font.size": 11,
            "legend.fontsize": 10,
            
            # Professional color scheme
            "axes.edgecolor": "#C0C0C0",
            "grid.color": "#E6E6E6",
            "grid.alpha": 0.6,
            
            # Clean layout
            "axes.spines.right": False, 
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            
            # Improved readability
            "font.family": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
        }
    )
    # Colorblind-friendly professional palette
    sns.set_palette("deep")

def currency_formatter():
    """Currency formatter for price axes."""
    return FuncFormatter(lambda x, pos: f"${x:,.0f}")

def k_formatter():
    """Thousands formatter for large numbers."""
    return FuncFormatter(lambda x, pos: f"${x/1000:.0f}K" if x >= 1000 else f"${x:.0f}")

def percentage_formatter():
    """Percentage formatter."""
    return FuncFormatter(lambda x, pos: f"{x:.1f}%")

def log_price(series):
    """Apply log1p transformation for price data."""
    return np.log1p(series)

def shade_splits(ax, train_end=2009, val_year=2010, test_year=2011):
    """
    Add temporal split shading to time series plots.
    Visually separates train/validation/test periods.
    """
    ax.axvspan(train_end-0.5, train_end+0.5, color="#B0BEC5", alpha=0.2, label="Train<=2009")
    ax.axvspan(val_year-0.5, val_year+0.5, color="#81D4FA", alpha=0.2, label="Val=2010")  
    ax.axvspan(test_year-0.5, test_year+0.5, color="#FFAB91", alpha=0.25, label="Test=2011")

def add_value_labels(ax, bars, format_func=None):
    """Add value labels on bar charts for better readability."""
    for bar in bars:
        height = bar.get_height()
        if format_func:
            label = format_func(height)
        else:
            label = f'{height:,.0f}'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9)

def create_confidence_ribbon(ax, x, y_median, y_lower, y_upper, color=None, alpha=0.2):
    """Create confidence interval ribbon for trend lines."""
    ax.fill_between(x, y_lower, y_upper, alpha=alpha, color=color, label='80% interval')

def setup_dual_axis_alternative(fig, data1, data2, labels, colors=None):
    """
    Alternative to dual-axis plots using synchronized subplots.
    More readable than twin axes.
    """
    if colors is None:
        colors = sns.color_palette("deep", 2)
    
    gs = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    return ax1, ax2

def apply_log_scale_safely(ax, axis='x', data=None):
    """Apply log scale only if data is suitable (all positive)."""
    if data is not None:
        if (data > 0).all():
            if axis == 'x':
                ax.set_xscale('log')
            else:
                ax.set_yscale('log')
        else:
            print(f"Warning: Cannot apply log scale to {axis}-axis due to non-positive values")

def set_currency_axis(ax, axis='y'):
    """Set proper currency formatting for price axes."""
    if axis == 'y':
        ax.yaxis.set_major_formatter(k_formatter())
        ax.set_ylabel('Price ($)')
    else:
        ax.xaxis.set_major_formatter(k_formatter())
        ax.set_xlabel('Price ($)')

def create_subplot_grid(nrows, ncols, figsize=None):
    """Create standardized subplot grid with proper spacing."""
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    plt.tight_layout(pad=3.0)
    return fig, axes

# Color constants for consistency
COLORS = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Orange accent
    'success': '#2ca02c',      # Green
    'warning': '#ff9500',      # Amber
    'danger': '#d62728',       # Red
    'info': '#17a2b8',         # Teal
    'light_gray': '#f8f9fa',   # Light background
    'dark_gray': '#6c757d',    # Dark text
}

def get_color_palette(n_colors=6):
    """Get consistent color palette for categorical data."""
    if n_colors <= 6:
        return list(COLORS.values())[:n_colors]
    else:
        # For more colors, use seaborn's palette
        return sns.color_palette("husl", n_colors)

# Export key functions and constants
__all__ = [
    'set_viz_theme',
    'currency_formatter', 
    'k_formatter',
    'percentage_formatter',
    'log_price',
    'shade_splits',
    'add_value_labels',
    'create_confidence_ribbon',
    'setup_dual_axis_alternative',
    'apply_log_scale_safely',
    'set_currency_axis',
    'create_subplot_grid',
    'get_color_palette',
    'COLORS'
]