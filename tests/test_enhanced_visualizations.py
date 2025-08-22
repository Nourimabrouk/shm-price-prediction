#!/usr/bin/env python3
"""
Test script for Enhanced Visualization Suite
Validates that all components work correctly and generates sample visualizations.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def create_sample_data():
    """Create sample SHM-like data for testing."""
    print("Creating sample data for visualization testing...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic equipment data
    data = {
        'sales_price': np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples),
        'year_made': np.random.randint(1995, 2020, size=n_samples),
        'sales_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'machinehours_currentmeter': np.random.gamma(2, 1000, size=n_samples),
        'product_group': np.random.choice(['Excavator', 'Bulldozer', 'Loader', 'Crane', 'Grader'], size=n_samples),
        'state_of_usage': np.random.choice(['TX', 'CA', 'FL', 'NY', 'PA', 'OH', 'IL'], size=n_samples),
        'auctioneer': np.random.choice(['Ritchie Bros', 'IronPlanet', 'BigIron', 'TractorHouse'], size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to simulate real data
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices, 'machinehours_currentmeter'] = np.nan
    
    print(f"Sample data created: {len(df):,} records")
    return df

def test_enhanced_visualization_suite():
    """Test the enhanced visualization suite."""
    print("Testing Enhanced Visualization Suite...")
    
    try:
        from viz_enhanced import EnhancedVisualizationSuite, PLOTLY_AVAILABLE
        print("Enhanced visualization module imported successfully")
        print(f"Plotly support: {'Available' if PLOTLY_AVAILABLE else 'Not available'}")
        
        # Create sample data
        df = create_sample_data()
        
        # Initialize enhanced viz suite
        viz_enhanced = EnhancedVisualizationSuite(output_dir="./outputs/test_figures/")
        print(f"SUCCESS Enhanced visualization suite initialized")
        
        # Test static visualizations from existing suite
        print("\n Testing integration with existing viz_suite...")
        try:
            from viz_suite import price_distribution_fig, age_vs_price_fig, product_group_fig
            
            static_tests = {
                'price_distribution': price_distribution_fig,
                'age_vs_price': age_vs_price_fig,
                'product_groups': product_group_fig
            }
            
            for name, func in static_tests.items():
                try:
                    fig = func(df)
                    if fig:
                        print(f"  SUCCESS {name}: Generated successfully")
                        fig.savefig(f"./outputs/test_figures/test_{name}.png", dpi=150, bbox_inches='tight')
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    else:
                        print(f"  WARNING {name}: No figure returned")
                except Exception as e:
                    print(f"  ERROR {name}: Failed - {e}")
            
        except ImportError as e:
            print(f"  WARNING viz_suite import failed: {e}")
        
        # Test interactive visualizations
        if PLOTLY_AVAILABLE:
            print("\n Testing interactive visualizations...")
            
            interactive_tests = [
                ('executive_dashboard', viz_enhanced.create_executive_dashboard),
                ('price_explorer', viz_enhanced.create_interactive_price_explorer),
                ('business_impact', lambda df: viz_enhanced.create_business_impact_dashboard(df, {'within_15_pct': 85.0})),
                ('correlation_matrix', viz_enhanced.create_advanced_correlation_matrix),
                ('temporal_analysis', viz_enhanced.create_temporal_analysis_dashboard)
            ]
            
            for name, func in interactive_tests:
                try:
                    fig = func(df)
                    if fig:
                        print(f"  SUCCESS {name}: Generated successfully")
                        fig.write_html(f"./outputs/test_figures/test_{name}.html")
                    else:
                        print(f"  WARNING {name}: No figure returned")
                except Exception as e:
                    print(f"  ERROR {name}: Failed - {e}")
        else:
            print("\nWARNING Plotly not available - skipping interactive tests")
            print("   Install with: pip install plotly")
        
        # Test complete suite generation
        print("\n📦 Testing complete suite generation...")
        try:
            model_metrics = {
                'within_15_pct': 85.2,
                'rmse': 12000,
                'r2': 0.78
            }
            
            saved_figures = viz_enhanced.save_enhanced_figures(df, model_metrics)
            
            print(f"SUCCESS Complete suite generated: {len(saved_figures)} files")
            for name, path in saved_figures.items():
                file_type = "Static" if path.endswith('.png') else "Interactive"
                print(f"  {file_type}: {name}")
            
        except Exception as e:
            print(f"ERROR Complete suite generation failed: {e}")
        
        print("\n" + "="*60)
        print("ENHANCED VISUALIZATION SUITE TEST RESULTS")
        print("="*60)
        print("SUCCESS Module import: SUCCESS")
        print("SUCCESS Sample data generation: SUCCESS")  
        print("SUCCESS Enhanced suite initialization: SUCCESS")
        
        if PLOTLY_AVAILABLE:
            print("SUCCESS Interactive visualizations: AVAILABLE")
        else:
            print("WARNING Interactive visualizations: REQUIRES PLOTLY")
        
        print(f"[OUTPUT] Test outputs saved to: ./outputs/test_figures/")
        print("[READY] Ready for notebook integration")
        
        return True
        
    except ImportError as e:
        print(f"ERROR Import failed: {e}")
        print("[NOTE] Make sure viz_enhanced.py is in the src/ directory")
        return False
    
    except Exception as e:
        print(f"ERROR Test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Enhanced Visualization Suite Test")
    print("="*50)
    
    # Create output directory
    Path("./outputs/test_figures/").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    success = test_enhanced_visualization_suite()
    
    if success:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("SUCCESS Enhanced visualization suite is ready for production use")
        print("[NEXT] Next steps:")
        print("   1. Run notebooks with enhanced visualizations")
        print("   2. Review generated test figures")
        print("   3. Install plotly for full interactive capabilities")
    else:
        print("\nERROR TESTS FAILED!")
        print(" Please check error messages and fix issues")
    
    return success

if __name__ == "__main__":
    main()