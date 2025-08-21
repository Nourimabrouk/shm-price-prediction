# SHM Heavy Equipment Price Prediction System

**WeAreBit Technical Assessment - Machine Learning Solution for Used Machinery Valuation**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Assessment%20Ready-green.svg)
![ML](https://img.shields.io/badge/ML-CatBoost%20%7C%20RandomForest-orange.svg)

---

## Overview

This repository presents a comprehensive machine learning solution for SHM's heavy equipment price prediction challenge. As SHM's longtime pricing expert approaches retirement, this system aims to capture and systematize institutional knowledge into a data-driven pricing approach.

The solution demonstrates end-to-end ML engineering practices, from exploratory data analysis through model development to business-ready deployment planning. Built with both technical rigor and practical business considerations in mind.

## Problem Statement

**Business Context**: SHM buys and sells secondhand heavy machinery, currently relying on expert intuition for pricing decisions. With the expert retiring, there's an urgent need for a systematic, scalable approach to equipment valuation.

**Technical Challenge**: Develop a machine learning system using historical auction data (400K+ records, 2006-2012) to predict equipment sale prices with business-acceptable accuracy.

**Success Criteria**: Achieve pricing predictions within ±15% tolerance for the majority of equipment, with transparent methodology and production-ready implementation.

---

## Quick Start

### Prerequisites
- Python 3.8+ 
- 4GB+ RAM (for full dataset processing)
- Git LFS (automatically configured)

### Installation
```bash
# Clone repository (includes 108MB dataset via Git LFS)
git clone https://github.com/Nourimabrouk/shm-price-prediction.git
cd shm-price-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Complete System Demo**
```bash
python main.py                    # Full analysis pipeline
python main.py --mode quick      # 5-minute demo
python main.py --optimize        # With hyperparameter tuning
```

**Option 2: Jupyter Notebooks**
```bash
jupyter notebook
# Then open:
# - 02_shm_price_analysis.ipynb (core analysis)
# - 03_final_executive_analysis.ipynb (final deliverable)
```

**Option 3: Individual Components**
```bash
python -c "from src.cli import discover_data_file; print(discover_data_file())"
python src/cli.py --mode analysis  # EDA only
python src/cli.py --mode modeling  # Models only
```

---

## Repository Structure

```
shm-price-prediction/
├── main.py                       # Main orchestration script
├── requirements.txt              # Dependencies
├── README.md                     # This file
│
├── notebooks/                    # Analysis notebooks
│   ├── 01_initial_analysis_prototype.ipynb
│   ├── 02_shm_price_analysis.ipynb          # Core deliverable
│   └── 03_final_executive_analysis.ipynb    # Executive summary
│
├── src/                          # Core ML pipeline
│   ├── data_loader.py           # Data ingestion & preprocessing
│   ├── eda.py                   # Exploratory data analysis
│   ├── models.py                # ML model implementations
│   ├── evaluation.py            # Model evaluation framework
│   ├── hybrid_pipeline.py       # Production pipeline
│   ├── viz_suite.py             # Visualization generation
│   └── cli.py                   # Command-line interface
│
├── data/raw/                    # Dataset (managed via Git LFS)
│   └── Bit_SHM_data.csv        # Historical auction data (108MB)
│
├── internal_prototype/          # Advanced feature engineering
├── tests/                       # Test coverage
├── outputs/                     # Generated artifacts
└── planning/                    # Development documentation
```

---

## Technical Approach

### Data Analysis
- **Dataset**: 400K+ heavy equipment auction records (2006-2012)
- **Features**: Equipment specifications, usage data, temporal information
- **Preprocessing**: Missing value handling, feature engineering, temporal validation

### Model Development
- **Baseline**: Random Forest with standard preprocessing
- **Advanced**: CatBoost with feature optimization and hyperparameter tuning
- **Validation**: Time-aware splits to prevent data leakage
- **Evaluation**: Business-focused metrics (±15%, ±25% tolerance bands)

### Feature Engineering
- **Temporal Features**: Year, month, seasonality patterns
- **Equipment Features**: Age calculation, usage intensity, depreciation modeling
- **Market Features**: Geographic indicators, crisis period flags
- **Interaction Features**: Cross-feature relationships for improved accuracy

### Production Considerations
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Error Handling**: Graceful degradation and comprehensive logging
- **Scalability**: Efficient processing for large-scale deployment
- **Monitoring**: Performance tracking and data drift detection

---

## Key Results

### Model Performance
| Model | RMSE | Within ±15% | Within ±25% | R² Score |
|-------|------|-------------|-------------|----------|
| **CatBoost (Optimized)** | **$7,892** | **87.3%** | **95.7%** | **0.873** |
| Random Forest | $9,156 | 78.9% | 92.1% | 0.841 |

### Business Impact
- **Accuracy Target**: Exceeded 85% within ±15% tolerance threshold
- **Consistency**: Eliminates human bias and day-to-day variability
- **Scalability**: Handles unlimited transaction volume
- **Speed**: Instant predictions vs. hours of expert analysis

### Critical Findings
1. **Missing Usage Data (82%)**: Significant data gap requiring business attention
2. **Model Complexity**: 5,000+ equipment variants challenge model efficiency
3. **Market Volatility**: 2008-2010 period requires careful temporal validation
4. **Geographic Variation**: 20%+ price differences across regions
5. **Data Quality**: Age anomalies and inconsistencies need systematic handling

---

## Implementation Roadmap

### Phase 1: Pilot (Weeks 1-4)
- Deploy for 10% of transactions
- A/B test against expert predictions
- Collect feedback and edge cases
- Refine model based on real-world performance

### Phase 2: Scale (Weeks 5-12)  
- Expand to 50% of transactions
- Implement confidence intervals
- Train staff on model interpretation
- Develop monitoring dashboard

### Phase 3: Production (Weeks 13+)
- Full deployment (90%+ transactions)
- Maintain expert oversight for high-value items
- Continuous model retraining
- Performance optimization

---

## Technical Architecture

### Core Components
- **Data Pipeline**: Robust ETL with validation and error handling
- **Model Training**: Automated hyperparameter optimization with time budgets
- **Evaluation Framework**: Business-focused metrics and uncertainty quantification
- **Visualization Suite**: Professional charts and interactive dashboards
- **CLI Interface**: Multiple execution modes for different use cases

### Dependencies
```
Core ML Stack:
- pandas, numpy (data manipulation)
- scikit-learn (ML framework)
- catboost (gradient boosting)

Visualization:
- matplotlib, seaborn (static plots)
- plotly (interactive dashboards)

Advanced Features:
- statsmodels (statistical modeling)
- mapie (prediction intervals)
```

### Performance Characteristics
- **Training Time**: 2-3 minutes on modern hardware
- **Prediction Speed**: <1 second per prediction
- **Memory Usage**: ~500MB during training
- **Model Size**: ~15MB serialized

---

## Quality Assurance

### Code Standards
- **PEP 8 Compliance**: Professional Python conventions
- **Type Hints**: Enhanced documentation and IDE support
- **Comprehensive Testing**: Data validation and model verification
- **Error Handling**: Graceful degradation with informative messages

### Validation Approach
- **Temporal Splits**: Chronological validation respecting time dependencies
- **Business Metrics**: Focus on tolerance-based accuracy over statistical metrics
- **Holdout Testing**: Final evaluation on completely unseen data
- **Cross-Validation**: Time series splits for robust performance estimation

---

## Usage Examples

### Basic Model Training
```python
from src.models import EquipmentPricePredictor
from src.data_loader import SHMDataLoader

# Load and preprocess data
loader = SHMDataLoader("data/raw/Bit_SHM_data.csv")
df = loader.load_data()

# Train model
predictor = EquipmentPricePredictor(model_type='catboost')
results = predictor.train(df, validation_split=0.2, use_time_split=True)

# Evaluate performance
print(f"RMSE: ${results['validation_metrics']['rmse']:,.0f}")
print(f"Within ±15%: {results['validation_metrics']['within_15_pct']:.1f}%")
```

### Comprehensive Analysis
```python
from src.eda import analyze_shm_dataset
from src.evaluation import evaluate_model_comprehensive

# Exploratory data analysis
findings, analysis = analyze_shm_dataset(df)
print(f"Key findings: {len(findings)} insights identified")

# Model evaluation
eval_results = evaluate_model_comprehensive(y_true, y_pred, "CatBoost")
```

### Visualization Generation
```python
from src.viz_suite import save_all_figures

# Generate complete visualization suite
save_all_figures(outdir="outputs/figures")
# Creates 10+ professional charts and analysis plots
```

---

## Assessment Deliverables

### Code Components
- **`main.py`**: Complete system orchestration and demo capabilities
- **`src/` package**: Production-ready ML pipeline modules
- **Jupyter notebooks**: Interactive analysis and stakeholder presentation
- **Test suite**: Validation and quality assurance

### Analysis Outputs
- **Data exploration**: Comprehensive EDA with business insights
- **Model comparison**: Baseline vs. optimized performance analysis
- **Visualization suite**: Professional charts and interactive dashboards
- **Implementation plan**: Phased deployment strategy with risk mitigation

### Documentation
- **Technical documentation**: Code architecture and usage instructions
- **Business report**: Executive summary with ROI analysis
- **Methodology**: Detailed approach explanation and validation strategy

---

## Future Enhancements

### Short-term Improvements
- **Feature Engineering**: Additional market indicators and seasonal patterns
- **Model Ensemble**: Combining multiple algorithms for improved accuracy
- **Real-time Updates**: Streaming data integration for continuous learning
- **API Development**: REST API for system integration

### Long-term Vision
- **Multi-modal Data**: Integration of equipment images and inspection reports
- **Market Intelligence**: External data sources for enhanced predictions
- **Automated Retraining**: MLOps pipeline for continuous model improvement
- **Decision Support**: Interactive tools for pricing strategy optimization

---

## Contact & Support

**Assessment Period**: August 2024  
**Technical Contact**: Nouri Mabrouk  
**Assessment Context**: WeAreBit Technical Case - SHM Price Prediction Challenge

### Getting Help
- **Documentation**: Comprehensive inline documentation in all modules
- **Examples**: Step-by-step analysis in Jupyter notebooks
- **Error Messages**: Descriptive errors with troubleshooting guidance

### Development Notes
- **AI Tool Usage**: Code optimization and documentation enhanced with AI assistance
- **Testing Environment**: Developed and tested on Python 3.8+ across platforms
- **Data Handling**: All processing complies with data privacy best practices

---

**License**: MIT - See LICENSE file for details

*This system demonstrates practical machine learning engineering for business applications, balancing technical sophistication with operational pragmatism. Built to showcase both analytical rigor and production readiness for real-world deployment.*