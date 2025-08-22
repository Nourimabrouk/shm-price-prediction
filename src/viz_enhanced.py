# viz_enhanced.py
# Next-Level Professional Visualization Suite for SHM Analysis
# Builds on viz_suite.py with interactive capabilities and advanced business intelligence

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Interactive visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARN] Plotly not available. Install with: pip install plotly")

# Import existing professional visualization infrastructure
try:
    from .viz_theme import (
        set_viz_theme, k_formatter, currency_formatter, percentage_formatter,
        shade_splits, log_price, add_value_labels, create_confidence_ribbon,
        apply_log_scale_safely, set_currency_axis, create_subplot_grid,
        get_color_palette, COLORS
    )
    
    # Import existing visualization functions to enhance
    from .viz_suite import (
        load_clean_df, price_distribution_fig, age_vs_price_fig,
        product_group_fig, temporal_trends_fig, usage_vs_price_fig,
        missingness_overview_fig, auctioneer_effect_fig, age_hours_joint_fig,
        state_premia_fig, temporal_heatmap_fig
    )
except ImportError:
    # Fallback for absolute imports
    from viz_theme import (
        set_viz_theme, k_formatter, currency_formatter, percentage_formatter,
        shade_splits, log_price, add_value_labels, create_confidence_ribbon,
        apply_log_scale_safely, set_currency_axis, create_subplot_grid,
        get_color_palette, COLORS
    )
    
    from viz_suite import (
        load_clean_df, price_distribution_fig, age_vs_price_fig,
        product_group_fig, temporal_trends_fig, usage_vs_price_fig,
        missingness_overview_fig, auctioneer_effect_fig, age_hours_joint_fig,
        state_premia_fig, temporal_heatmap_fig
    )

class EnhancedVisualizationSuite:
    """
    Next-level professional visualization suite combining static and interactive visualizations
    with advanced business intelligence features.
    """
    
    def __init__(self, output_dir: str = "./outputs/figures/", dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.plotly_theme = "plotly_white"
        # Use Plotly color sequence when available; otherwise provide a safe fallback
        if PLOTLY_AVAILABLE:
            self.color_sequence = px.colors.qualitative.Set2
        else:
            self.color_sequence = [
                "#1f77b4", "#ff7f0e", "#2ca02c",
                "#d62728", "#9467bd", "#8c564b"
            ]
        
        # Set consistent themes
        set_viz_theme()
        
        if PLOTLY_AVAILABLE:
            # Set Plotly theme for consistency
            import plotly.io as pio
            pio.templates.default = self.plotly_theme
    
    def create_executive_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates an interactive executive dashboard with key business metrics.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly required for interactive dashboard")
            return None
        
        # Prepare data
        df_clean = df.dropna(subset=['sales_price', 'year_made', 'sales_date'])
        df_clean['age_years'] = 2024 - df_clean['year_made']
        df_clean['price_k'] = df_clean['sales_price'] / 1000
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Price vs Age Trend', 
                           'Market Volume Trend', 'Geographic Distribution'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Price Distribution (Histogram with Box Plot)
        fig.add_trace(
            go.Histogram(
                x=df_clean['price_k'],
                nbinsx=50,
                name='Price Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add box plot overlay
        fig.add_trace(
            go.Box(
                y=df_clean['price_k'],
                name='Price Range',
                marker_color='darkblue',
                boxmean=True
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Price vs Age Scatter with Trendline
        sample_size = min(5000, len(df_clean))
        df_sample = df_clean.sample(n=sample_size, random_state=42)
        
        fig.add_trace(
            go.Scatter(
                x=df_sample['age_years'],
                y=df_sample['price_k'],
                mode='markers',
                name='Equipment Sales',
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=df_sample['price_k'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Price ($K)")
                ),
                hovertemplate='Age: %{x} years<br>Price: $%{y}K<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add trendline
        z = np.polyfit(df_sample['age_years'], df_sample['price_k'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_sample['age_years'].min(), df_sample['age_years'].max(), 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Depreciation Trend',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        # 3. Market Volume Trend
        monthly_volume = (df_clean.groupby(df_clean['sales_date'].dt.to_period('M'))
                         .size().reset_index())
        monthly_volume['date'] = monthly_volume['sales_date'].astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=monthly_volume['date'],
                y=monthly_volume[0],
                mode='lines+markers',
                name='Monthly Volume',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 4. Geographic Distribution (Top States)
        if 'state_of_usage' in df_clean.columns:
            state_counts = df_clean['state_of_usage'].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(
                    x=state_counts.values,
                    y=state_counts.index,
                    orientation='h',
                    name='Sales by State',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>SHM Heavy Equipment Market Dashboard</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=False,
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Price ($K)", row=1, col=1)
        fig.update_xaxes(title_text="Equipment Age (years)", row=1, col=2)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_xaxes(title_text="Number of Sales", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Price ($K)", row=1, col=2)
        fig.update_yaxes(title_text="Sales Volume", row=2, col=1)
        fig.update_yaxes(title_text="State", row=2, col=2)
        
        return fig
    
    def create_interactive_price_explorer(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates an interactive price exploration tool with filters and drill-down capabilities.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly required for interactive explorer")
            return None
        
        # Prepare data
        df_clean = df.dropna(subset=['sales_price', 'year_made'])
        df_clean['age_years'] = 2024 - df_clean['year_made']
        df_clean['price_k'] = df_clean['sales_price'] / 1000
        
        # Sample for performance if dataset is large
        if len(df_clean) > 10000:
            df_plot = df_clean.sample(n=10000, random_state=42)
        else:
            df_plot = df_clean
        
        # Create main scatter plot
        fig = px.scatter(
            df_plot,
            x='age_years',
            y='price_k',
            color='product_group' if 'product_group' in df_plot.columns else None,
            size='machinehours_currentmeter' if 'machinehours_currentmeter' in df_plot.columns else None,
            hover_data={
                'sales_date': True,
                'auctioneer': True if 'auctioneer' in df_plot.columns else False,
                'state_of_usage': True if 'state_of_usage' in df_plot.columns else False
            },
            title="<b>Interactive Equipment Price Explorer</b>",
            labels={
                'age_years': 'Equipment Age (years)',
                'price_k': 'Sale Price ($K)',
                'product_group': 'Product Group'
            },
            template=self.plotly_theme,
            height=700
        )
        
        # Add trendline
        fig.add_traces(px.scatter(df_plot, x='age_years', y='price_k', trendline='lowess').data[1:])
        
        # Update layout for better interactivity
        fig.update_layout(
            title=dict(x=0.5, font=dict(size=18)),
            xaxis=dict(title=dict(font=dict(size=14))),
            yaxis=dict(title=dict(font=dict(size=14))),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Add range slider for better navigation
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        return fig
    
    def create_business_impact_dashboard(self, df: pd.DataFrame, 
                                       model_metrics: dict = None) -> go.Figure:
        """
        Creates a business impact dashboard showing financial implications of pricing accuracy.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly required for business dashboard")
            return None
        
        # Prepare data
        df_clean = df.dropna(subset=['sales_price'])
        
        # Calculate business metrics
        total_volume = len(df_clean)
        total_value = df_clean['sales_price'].sum()
        avg_price = df_clean['sales_price'].mean()
        
        # Price risk analysis
        high_value_threshold = 100000
        high_value_count = (df_clean['sales_price'] > high_value_threshold).sum()
        high_value_percentage = (high_value_count / total_volume) * 100
        
        # Market segmentation
        price_bands = pd.cut(df_clean['sales_price'], 
                           bins=[0, 20000, 50000, 100000, np.inf],
                           labels=['Budget', 'Mid-range', 'Premium', 'Ultra-premium'])
        segment_stats = price_bands.value_counts()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Market Size Metrics', 'Risk Distribution', 
                           'Market Segmentation', 'Accuracy Impact'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Key Performance Indicators
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=total_value / 1e6,
                delta={"reference": (total_value * 0.9) / 1e6},
                title={"text": "Total Market Value<br>(Million $)"},
                number={'prefix': "$", 'suffix': "M"},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # 2. Risk Distribution (High vs Low Value)
        risk_labels = ['Standard Risk (<$100K)', 'High Risk (>=$100K)']
        risk_values = [total_volume - high_value_count, high_value_count]
        
        fig.add_trace(
            go.Pie(
                labels=risk_labels,
                values=risk_values,
                hole=0.4,
                marker_colors=['lightblue', 'red'],
                textinfo='label+percent'
            ),
            row=1, col=2
        )
        
        # 3. Market Segmentation
        fig.add_trace(
            go.Bar(
                x=segment_stats.index,
                y=segment_stats.values,
                marker_color=['green', 'blue', 'orange', 'red'],
                text=segment_stats.values,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Accuracy Impact Simulation
        if model_metrics:
            accuracy_levels = [60, 70, 80, 85, 90, 95]
            potential_savings = []
            
            for acc in accuracy_levels:
                # Estimate potential savings based on pricing accuracy
                base_error_rate = 0.15  # 15% baseline error
                current_error_rate = (100 - acc) / 100
                error_reduction = max(0, base_error_rate - current_error_rate)
                savings = total_value * error_reduction * 0.1  # 10% of error reduction as savings
                potential_savings.append(savings / 1e6)  # Convert to millions
            
            fig.add_trace(
                go.Scatter(
                    x=accuracy_levels,
                    y=potential_savings,
                    mode='lines+markers',
                    name='Potential Savings',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=2
            )
            
            # Add current model performance if available
            if 'within_15_pct' in model_metrics:
                current_acc = model_metrics['within_15_pct']
                current_savings = np.interp(current_acc, accuracy_levels, potential_savings)
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_acc],
                        y=[current_savings],
                        mode='markers',
                        name='Current Model',
                        marker=dict(size=15, color='red', symbol='star')
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Business Impact Analysis Dashboard</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Market Segment", row=2, col=1)
        fig.update_xaxes(title_text="Accuracy (%)", row=2, col=2)
        fig.update_yaxes(title_text="Number of Sales", row=2, col=1)
        fig.update_yaxes(title_text="Potential Savings ($M)", row=2, col=2)
        
        return fig
    
    def create_advanced_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates an interactive correlation matrix with advanced features.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly required for interactive correlation matrix")
            return None
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="<b>Interactive Feature Correlation Matrix</b>",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_temporal_analysis_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates an advanced temporal analysis dashboard with multiple time views.
        """
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly required for temporal dashboard")
            return None
        
        # Prepare temporal data
        df_temporal = df.dropna(subset=['sales_date', 'sales_price']).copy()
        df_temporal['year'] = df_temporal['sales_date'].dt.year
        df_temporal['month'] = df_temporal['sales_date'].dt.month
        df_temporal['quarter'] = df_temporal['sales_date'].dt.quarter
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Price Trends', 'Monthly Seasonality',
                           'Quarterly Volume', 'Market Events'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Annual trends with volume overlay
        annual_stats = df_temporal.groupby('year').agg({
            'sales_price': ['mean', 'median', 'count']
        }).round(0)
        annual_stats.columns = ['mean_price', 'median_price', 'volume']
        
        fig.add_trace(
            go.Scatter(
                x=annual_stats.index,
                y=annual_stats['median_price'],
                mode='lines+markers',
                name='Median Price',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=annual_stats.index,
                y=annual_stats['volume'],
                name='Annual Volume',
                marker_color='lightblue',
                opacity=0.6
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Monthly seasonality
        monthly_stats = df_temporal.groupby('month')['sales_price'].median()
        
        fig.add_trace(
            go.Bar(
                x=monthly_stats.index,
                y=monthly_stats.values,
                name='Monthly Median Price',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # 3. Quarterly volume trends
        quarterly_volume = df_temporal.groupby(['year', 'quarter']).size().reset_index()
        quarterly_volume['period'] = quarterly_volume['year'].astype(str) + '-Q' + quarterly_volume['quarter'].astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=quarterly_volume['period'],
                y=quarterly_volume[0],
                mode='lines+markers',
                name='Quarterly Volume',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # 4. Market events (price volatility)
        monthly_volatility = df_temporal.groupby(df_temporal['sales_date'].dt.to_period('M'))['sales_price'].std()
        monthly_volatility.index = monthly_volatility.index.astype(str)
        
        fig.add_trace(
            go.Scatter(
                x=monthly_volatility.index,
                y=monthly_volatility.values,
                mode='lines+markers',
                name='Price Volatility',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="<b>Temporal Market Analysis Dashboard</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_enhanced_figures(self, df: pd.DataFrame, model_metrics: dict = None):
        """
        Generates and saves all enhanced visualizations.
        """
        print("[VIZ] Generating Enhanced Professional Visualizations...")
        
        saved_figures = {}
        
        # Static visualizations from original suite
        print("[DATA] Creating static professional visualizations...")
        static_figures = {
            'price_distribution': price_distribution_fig(df),
            'age_vs_price': age_vs_price_fig(df),
            'product_groups': product_group_fig(df),
            'temporal_trends': temporal_trends_fig(df),
            'usage_vs_price': usage_vs_price_fig(df),
            'missingness': missingness_overview_fig(df),
            'auctioneer_effect': auctioneer_effect_fig(df),
            'age_hours_joint': age_hours_joint_fig(df),
            'state_premia': state_premia_fig(df),
            'temporal_heatmap': temporal_heatmap_fig(df)
        }
        
        for name, fig in static_figures.items():
            if fig is not None:
                filepath = self.output_dir / f"{name}_professional.png"
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                saved_figures[name] = str(filepath)
                plt.close(fig)
        
        # Interactive visualizations
        if PLOTLY_AVAILABLE:
            print("[START] Creating interactive visualizations...")
            
            # Executive Dashboard
            exec_dashboard = self.create_executive_dashboard(df)
            if exec_dashboard:
                exec_path = self.output_dir / "executive_dashboard.html"
                exec_dashboard.write_html(str(exec_path))
                saved_figures['executive_dashboard'] = str(exec_path)
            
            # Price Explorer
            price_explorer = self.create_interactive_price_explorer(df)
            if price_explorer:
                explorer_path = self.output_dir / "price_explorer.html"
                price_explorer.write_html(str(explorer_path))
                saved_figures['price_explorer'] = str(explorer_path)
            
            # Business Impact Dashboard
            business_dashboard = self.create_business_impact_dashboard(df, model_metrics)
            if business_dashboard:
                business_path = self.output_dir / "business_impact_dashboard.html"
                business_dashboard.write_html(str(business_path))
                saved_figures['business_dashboard'] = str(business_path)
            
            # Correlation Matrix
            corr_matrix = self.create_advanced_correlation_matrix(df)
            if corr_matrix:
                corr_path = self.output_dir / "correlation_matrix.html"
                corr_matrix.write_html(str(corr_path))
                saved_figures['correlation_matrix'] = str(corr_path)
            
            # Temporal Dashboard
            temporal_dashboard = self.create_temporal_analysis_dashboard(df)
            if temporal_dashboard:
                temporal_path = self.output_dir / "temporal_dashboard.html"
                temporal_dashboard.write_html(str(temporal_path))
                saved_figures['temporal_dashboard'] = str(temporal_path)
        
        else:
            print("[WARN] Plotly not available - interactive visualizations skipped")
        
        print(f"[OK] Enhanced visualizations saved to: {self.output_dir}")
        print(f"[DATA] Total files generated: {len(saved_figures)}")
        
        return saved_figures

def create_notebook_visualization_cell(cell_type: str = "executive") -> str:
    """
    Returns ready-to-use notebook code for different visualization types.
    """
    
    if cell_type == "executive":
        return '''
# Executive Dashboard - Interactive Business Overview
from src.viz_enhanced import EnhancedVisualizationSuite

# Initialize enhanced visualization suite
viz_enhanced = EnhancedVisualizationSuite(output_dir="./outputs/figures/")

# Create interactive executive dashboard
exec_dashboard = viz_enhanced.create_executive_dashboard(df)
if exec_dashboard:
    exec_dashboard.show()
else:
    print("[WARN] Install plotly for interactive dashboards: pip install plotly")
'''
    
    elif cell_type == "explorer":
        return '''
# Interactive Price Explorer - Deep Dive Analysis
price_explorer = viz_enhanced.create_interactive_price_explorer(df)
if price_explorer:
    price_explorer.show()
'''
    
    elif cell_type == "business":
        return '''
# Business Impact Analysis
# Include model metrics if available
model_metrics = {
    'within_15_pct': 85.2,  # Replace with actual metrics
    'rmse': 12000,
    'r2': 0.78
}

business_dashboard = viz_enhanced.create_business_impact_dashboard(df, model_metrics)
if business_dashboard:
    business_dashboard.show()
'''
    
    elif cell_type == "complete":
        return '''
# Generate Complete Enhanced Visualization Suite
print("[VIZ] Generating Complete Enhanced Visualization Suite...")

# Save all enhanced figures (static + interactive)
saved_figures = viz_enhanced.save_enhanced_figures(df, model_metrics=model_metrics)

print("\\n[DATA] Generated Visualizations:")
for name, path in saved_figures.items():
    print(f"  [OK] {name}: {path}")

print("\\n[START] Interactive dashboards can be opened in browser from HTML files")
print("[DATA] Static plots saved as high-resolution PNG files")
'''
    
    return "# Unknown cell type"

# Export key components
__all__ = [
    'EnhancedVisualizationSuite',
    'create_notebook_visualization_cell',
    'PLOTLY_AVAILABLE'
]
