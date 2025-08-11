# Enhanced Phenology Analysis for Dominican Republic Rice Cultivation

## Overview

This project provides an enhanced phenology analysis framework for rice cultivation studies in the Dominican Republic using MODIS NDVI data and Google Earth Engine. The analysis includes comprehensive phenology metrics, validation methods, and advanced visualizations.

## Key Features

### 🌱 Enhanced Phenology Metrics
- **Start of Season (SOS)**: When vegetation growth begins
- **End of Season (EOS)**: When vegetation growth ends
- **Length of Season (LOS)**: Duration of growing season
- **Seasonality Index**: Measure of seasonal variation strength
- **Growth Rate**: Rate of vegetation development
- **Annual Amplitude**: Primary seasonal cycle strength
- **Semi-annual Amplitude**: Secondary seasonal cycle strength
- **Peak NDVI Value**: Maximum vegetation index during season
- **Variability Index**: Measure of inter-annual variability

### 🔍 Validation & Quality Control
- **Range Validation**: Ensures metrics are within expected ranges
- **Logical Consistency**: Checks for logical relationships between metrics
- **Statistical Significance**: Tests for meaningful patterns
- **Outlier Detection**: Identifies and handles anomalous data
- **Confidence Intervals**: Provides uncertainty estimates
- **Data Completeness**: Assesses data quality and coverage

### 📊 Advanced Visualizations
- **Interactive Maps**: Google Earth Engine-based spatial visualizations
- **Statistical Plots**: Distribution analysis and correlation matrices
- **Time Series Analysis**: Temporal pattern identification
- **Quality Assessment**: Data quality and validation summaries

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Earth Engine account
- Google Drive access (for data export)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd phenology-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Authenticate with Google Earth Engine**:
```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Usage

### Quick Start

```python
from enhanced_phenology_analysis import EnhancedPhenologyAnalysis

# Initialize analysis
analysis = EnhancedPhenologyAnalysis('Dominican Republic', 2011, 2022)

# Run complete analysis
analysis.setup_study_area()
analysis.get_enhanced_modis_data()
analysis.extract_enhanced_phenology()
analysis.create_enhanced_rice_mask()
analysis.validate_phenology_metrics()

# Create visualizations
analysis.create_visualizations()
analysis.create_statistical_plots()
analysis.create_correlation_analysis()

# Generate summary report
analysis.generate_summary_report()
```

### Step-by-Step Analysis

#### 1. Data Collection
```python
# Get MODIS NDVI data with quality filtering
modis_data = analysis.get_enhanced_modis_data()
```

#### 2. Phenology Extraction
```python
# Extract comprehensive phenology metrics
phenology_results = analysis.extract_enhanced_phenology()
```

#### 3. Rice Masking
```python
# Create enhanced rice mask using multiple criteria
rice_mask = analysis.create_enhanced_rice_mask()
```

#### 4. Validation
```python
# Validate results with statistical tests
validation_results, sample_data = analysis.validate_phenology_metrics()
```

#### 5. Visualization
```python
# Create interactive maps
phenology_map = analysis.create_visualizations()

# Create statistical plots
stats_fig = analysis.create_statistical_plots()

# Create correlation analysis
corr_matrix = analysis.create_correlation_analysis()
```

## Methodology

### Harmonic Analysis
The analysis uses harmonic regression to fit annual and semi-annual cycles to NDVI time series:

```
NDVI(t) = A₀ + A₁cos(ω₁t) + B₁sin(ω₁t) + A₂cos(ω₂t) + B₂sin(ω₂t)
```

Where:
- ω₁ = 2π/365.25 (annual cycle)
- ω₂ = 4π/365.25 (semi-annual cycle)

### Phenology Metrics Calculation

#### Start of Season (SOS)
- Extracted from the phase of the annual harmonic component
- Represents the timing of vegetation growth initiation

#### End of Season (EOS)
- Calculated as SOS + 180 days (typical growing season length)
- Can be adjusted based on local agricultural practices

#### Seasonality Index
- Ratio of annual amplitude to total amplitude
- Measures the strength of seasonal variation

#### Growth Rate
- Derived from the amplitude and frequency of the annual cycle
- Indicates the rate of vegetation development

### Rice Masking Criteria
1. **Land Cover**: IGBP classes 12 (Croplands) and 14 (Cropland/Natural Vegetation Mosaics)
2. **NDVI Range**: 0.2-0.9 (typical for rice cultivation)
3. **Elevation**: Below 1000m (rice typically grown in lowlands)

## Output Files

### Maps and Visualizations
- Interactive Google Earth Engine maps
- Statistical distribution plots
- Correlation matrices
- Time series visualizations

### Data Exports
- Phenology metrics as GeoTIFF files
- Rice mask as GeoTIFF file
- Statistical summaries as CSV files
- Validation reports as text files

### Summary Reports
- Comprehensive analysis summary
- Quality assessment results
- Key findings and recommendations

## Validation Methods

### Range Validation
- SOS: 0-365 days
- EOS: 0-365 days
- LOS: 30-365 days
- Seasonality: 0-1

### Logical Consistency
- SOS < EOS
- LOS > 0
- Seasonality within valid range

### Statistical Validation
- Confidence intervals (95%)
- Outlier detection (IQR method)
- Correlation analysis
- Data completeness assessment

## Example Results

### Typical Phenology Patterns in Dominican Republic Rice Areas
- **Average SOS**: ~120 days (late April)
- **Average EOS**: ~300 days (late October)
- **Average LOS**: ~180 days
- **Seasonality Index**: 0.6-0.8 (strong seasonal patterns)

### Quality Metrics
- **Data Completeness**: >90%
- **Outlier Percentage**: <5%
- **Validation Pass Rate**: >95%

## Advanced Features

### Custom Analysis
```python
# Custom study area
analysis = EnhancedPhenologyAnalysis('Haiti', 2015, 2020)

# Custom validation parameters
validation_results = analysis.validate_phenology_metrics(sample_size=2000)

# Custom export settings
export_tasks = analysis.export_results(export_to_drive=True)
```

### Integration with STICS Model
The analysis can be integrated with the STICS crop model for:
- Model parameterization
- Validation of simulated phenology
- Climate impact assessment

## Troubleshooting

### Common Issues

1. **Google Earth Engine Authentication**
   ```python
   # Re-authenticate if needed
   ee.Authenticate()
   ee.Initialize()
   ```

2. **Memory Issues**
   - Reduce sample size in validation
   - Use smaller study areas
   - Process data in chunks

3. **Data Quality Issues**
   - Check MODIS data availability
   - Verify study area boundaries
   - Review quality filtering parameters

### Performance Optimization
- Use appropriate sample sizes for validation
- Optimize Earth Engine queries
- Cache intermediate results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{enhanced_phenology_analysis,
  title={Enhanced Phenology Analysis for Rice Cultivation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/phenology-analysis}
}
```

## Contact

For questions or support, please contact:
- Email: your.email@example.com
- GitHub Issues: [Repository Issues](https://github.com/your-repo/phenology-analysis/issues)

## Acknowledgments

- Google Earth Engine team for satellite data access
- MODIS team for vegetation index products
- Scientific community for phenology analysis methods
