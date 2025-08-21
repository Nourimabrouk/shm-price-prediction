"""Professional visualization system for SHM equipment price analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class SHMPlotGenerator:
    """Generate professional visualizations for SHM equipment data analysis."""
    
    def __init__(self, output_dir: str = "./plots/"):
        """Initialize plot generator with styling."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set professional color scheme
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Deep magenta
            'accent': '#F18F01',       # Orange accent
            'danger': '#C73E1D',       # Red for warnings
            'success': '#4CAF50',      # Green for success
            'light': '#E8F4FD',        # Light blue
            'dark': '#1B365D'          # Dark blue
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_price_distribution(self, df: pd.DataFrame, save: bool = True) -> str:
        """Create comprehensive price distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SHM Equipment Price Distribution Analysis', fontsize=16, fontweight='bold')
        
        prices = df['sales_price'].dropna()
        
        # 1. Basic distribution
        ax1 = axes[0, 0]
        ax1.hist(prices, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax1.axvline(prices.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ${prices.mean():,.0f}')
        ax1.axvline(prices.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: ${prices.median():,.0f}')
        
        ax1.set_xlabel('Sales Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Price Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Log-scale distribution
        ax2 = axes[0, 1]
        log_prices = np.log1p(prices)
        ax2.hist(log_prices, bins=50, alpha=0.7, color=self.colors['secondary'], edgecolor='black')
        ax2.set_xlabel('Log(Sales Price + 1)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Log-Transformed Price Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot by price ranges
        ax3 = axes[1, 0]
        price_bands = pd.cut(prices, 
                           bins=[0, 20000, 50000, 100000, np.inf],
                           labels=['<$20K', '$20-50K', '$50-100K', '>$100K'])
        
        price_band_data = [prices[price_bands == band].values for band in price_bands.cat.categories]
        price_band_labels = [f'{label}\n(n={len(data)})' for label, data in zip(price_bands.cat.categories, price_band_data)]
        
        bp = ax3.boxplot(price_band_data, labels=price_band_labels, patch_artist=True)
        
        # Color boxes
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['danger']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Sales Price ($)')
        ax3.set_title('Price Distribution by Ranges')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_data = [
            ['Count', f'{len(prices):,}'],
            ['Mean', f'${prices.mean():,.0f}'],
            ['Median', f'${prices.median():,.0f}'],
            ['Std Dev', f'${prices.std():,.0f}'],
            ['Min', f'${prices.min():,.0f}'],
            ['Max', f'${prices.max():,.0f}']
        ]
        
        table = ax4.table(cellText=stats_data,
                         colLabels=['Statistic', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Price Statistics Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / 'price_distribution_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return "Plot displayed"


def create_all_eda_plots(df: pd.DataFrame, key_findings: List[Dict], 
                         output_dir: str = "./plots/") -> Dict[str, str]:
    """Generate all EDA visualizations for the SHM dataset."""
    plotter = SHMPlotGenerator(output_dir)
    
    plots = {
        'price_distribution': plotter.plot_price_distribution(df)
    }
    
    print(f"\nAll EDA plots generated successfully!")
    for plot_name, plot_path in plots.items():
        print(f"  {plot_name}: {plot_path}")
    
    return plots


if __name__ == "__main__":
    print("✓ Plots module loaded successfully")
