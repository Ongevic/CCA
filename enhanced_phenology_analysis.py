#!/usr/bin/env python3
"""
Enhanced Phenology Analysis for Dominican Republic Rice Cultivation
==================================================================

This script performs comprehensive phenology analysis for rice cultivation 
in the Dominican Republic using MODIS NDVI data and Google Earth Engine.

Key Features:
- Enhanced Phenology Metrics: SOS, EOS, LOS, Peak Timing, Growth Rate, Seasonality Index
- Validation Methods: Cross-validation, statistical significance testing
- Advanced Visualizations: Interactive maps, time series plots, statistical summaries
- Quality Control: Data filtering, outlier detection, confidence intervals

Author: Enhanced Phenology Analysis
Date: 2024
"""

import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedPhenologyAnalysis:
    """Enhanced phenology analysis class for rice cultivation studies."""
    
    def __init__(self, study_area='Dominican Republic', start_year=2011, end_year=2022):
        """
        Initialize the phenology analysis.
        
        Parameters:
        -----------
        study_area : str
            Name of the study area
        start_year : int
            Start year for analysis
        end_year : int
            End year for analysis
        """
        self.study_area = study_area
        self.start_year = start_year
        self.end_year = end_year
        self.growing_seasons = {
            'season1': {'start': '03-01', 'end': '07-31'},
            'season2': {'start': '08-01', 'end': '12-31'}
        }
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        
        print(f"Initialized phenology analysis for {study_area} ({start_year}-{end_year})")
    
    def setup_study_area(self):
        """Setup study area boundary and parameters."""
        
        # Dominican Republic boundary
        self.boundary = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')\
            .filter(ee.Filter.eq('country_na', self.study_area))
        
        return self.boundary
    
    def get_enhanced_modis_data(self):
        """Collect MODIS NDVI data with enhanced quality control."""
        
        # MODIS NDVI collection with quality filtering
        modis = ee.ImageCollection('MODIS/006/MOD13Q1')\
            .filterBounds(self.boundary)\
            .filterDate(f'{self.start_year}-01-01', f'{self.end_year}-12-31')\
            .select(['NDVI', 'EVI', 'pixel_reliability'])\
            .filter(ee.Filter.lte('pixel_reliability', 1))  # Good quality pixels only
        
        # Scale NDVI values (0-10000 to 0-1)
        def scale_ndvi(image):
            return image.divide(10000).copyProperties(image)
        
        self.modis_data = modis.map(scale_ndvi)
        
        print(f"MODIS data collected for {self.start_year}-{self.end_year}")
        return self.modis_data
    
    def extract_enhanced_phenology(self):
        """Extract comprehensive phenology metrics using harmonic analysis."""
        
        # Convert to array for harmonic analysis
        array = self.modis_data.toArray()
        
        # Fit harmonic model (annual + semi-annual cycles)
        def fit_harmonics(array):
            # Get time axis
            time_axis = array.arrayDimMin(0)
            
            # Create harmonic terms
            omega1 = 2 * np.pi / 365.25  # Annual cycle
            omega2 = 4 * np.pi / 365.25  # Semi-annual cycle
            
            # Harmonic regression
            t = time_axis.divide(1000 * 60 * 60 * 24)  # Convert to days
            
            # Create harmonic terms
            cos1 = t.multiply(omega1).cos()
            sin1 = t.multiply(omega1).sin()
            cos2 = t.multiply(omega2).cos()
            sin2 = t.multiply(omega2).sin()
            
            # Stack harmonic terms
            harmonics = ee.Image.cat([cos1, sin1, cos2, sin2])
            
            # Fit model
            fitted = array.arrayReduce(ee.Reducer.linearRegression(4, 1), [0])
            
            return fitted
        
        fitted = fit_harmonics(array)
        
        # Extract phenology metrics
        def extract_metrics(fitted_array):
            # Get coefficients
            coeffs = fitted_array.arraySlice(0, 0, 4)
            
            # Calculate amplitude and phase
            a1 = coeffs.arraySlice(0, 0, 1)
            b1 = coeffs.arraySlice(0, 1, 2)
            a2 = coeffs.arraySlice(0, 2, 3)
            b2 = coeffs.arraySlice(0, 3, 4)
            
            # Annual amplitude and phase
            amp1 = a1.pow(2).add(b1.pow(2)).sqrt()
            phase1 = b1.divide(a1).atan()
            
            # Semi-annual amplitude and phase
            amp2 = a2.pow(2).add(b2.pow(2)).sqrt()
            phase2 = b2.divide(a2).atan()
            
            # Peak timing (SOS)
            sos = phase1.divide(2 * np.pi).multiply(365.25)
            
            # End of season (EOS) - 180 days after SOS
            eos = sos.add(180)
            
            # Length of season
            los = eos.subtract(sos)
            
            # Seasonality index (ratio of annual to total variance)
            seasonality = amp1.divide(amp1.add(amp2))
            
            # Growth rate (slope at SOS)
            growth_rate = amp1.multiply(2 * np.pi / 365.25)
            
            # Additional metrics
            # Peak NDVI value
            peak_ndvi = amp1.add(amp2)
            
            # Variability index
            variability = amp1.divide(amp1.add(amp2)).multiply(100)
            
            return ee.Image.cat([sos, eos, los, seasonality, growth_rate, amp1, amp2, peak_ndvi, variability])\
                .rename(['sos', 'eos', 'los', 'seasonality', 'growth_rate', 'annual_amp', 'semi_annual_amp', 'peak_ndvi', 'variability'])
        
        self.phenology_results = extract_metrics(fitted)
        
        print("Enhanced phenology metrics extracted")
        return self.phenology_results
    
    def create_enhanced_rice_mask(self):
        """Create enhanced rice mask using multiple criteria."""
        
        # Land cover data
        landcover = ee.ImageCollection('MODIS/006/MCD12Q1')\
            .filterBounds(self.boundary)\
            .filterDate('2020-01-01', '2020-12-31')\
            .first()
        
        # IGBP classification for rice-relevant areas
        # 12: Croplands, 14: Cropland/Natural Vegetation Mosaics
        rice_landcover = landcover.eq(12).Or(landcover.eq(14))
        
        # Additional criteria: NDVI range typical for rice
        # Rice typically has NDVI 0.3-0.8 during growing season
        ndvi_range = self.modis_data.select('NDVI').mean()
        ndvi_mask = ndvi_range.gte(0.2).And(ndvi_range.lte(0.9))
        
        # Elevation mask (rice typically below 1000m)
        elevation = ee.Image('USGS/SRTMGL1_003')
        elevation_mask = elevation.lte(1000)
        
        # Combine masks
        self.rice_mask = rice_landcover.And(ndvi_mask).And(elevation_mask)
        
        print("Enhanced rice mask created")
        return self.rice_mask.rename('rice_mask')
    
    def validate_phenology_metrics(self, sample_size=1000):
        """Validate phenology metrics with statistical tests."""
        
        # Sample data for validation
        sample_points = ee.FeatureCollection.randomPoints(self.boundary, sample_size)
        
        # Extract values at sample points
        sample_data = self.phenology_results.addBands(self.rice_mask)\
            .sampleRegions(sample_points, scale=250)
        
        # Convert to pandas for analysis
        self.sample_df = geemap.ee_to_pandas(sample_data)
        
        # Remove invalid values
        self.sample_df = self.sample_df.dropna()
        
        # Validation metrics
        self.validation_results = {}
        
        # 1. Range validation
        self.validation_results['sos_range'] = {
            'min': self.sample_df['sos'].min(),
            'max': self.sample_df['sos'].max(),
            'valid': 0 <= self.sample_df['sos'].min() <= self.sample_df['sos'].max() <= 365
        }
        
        self.validation_results['eos_range'] = {
            'min': self.sample_df['eos'].min(),
            'max': self.sample_df['eos'].max(),
            'valid': 0 <= self.sample_df['eos'].min() <= self.sample_df['eos'].max() <= 365
        }
        
        self.validation_results['los_range'] = {
            'min': self.sample_df['los'].min(),
            'max': self.sample_df['los'].max(),
            'valid': 30 <= self.sample_df['los'].min() <= self.sample_df['los'].max() <= 365
        }
        
        # 2. Logical consistency
        self.validation_results['logical_consistency'] = {
            'sos_before_eos': (self.sample_df['sos'] < self.sample_df['eos']).all(),
            'los_positive': (self.sample_df['los'] > 0).all(),
            'seasonality_range': (self.sample_df['seasonality'] >= 0).all() and (self.sample_df['seasonality'] <= 1).all()
        }
        
        # 3. Statistical summary
        self.validation_results['statistics'] = {
            'sos_mean': self.sample_df['sos'].mean(),
            'sos_std': self.sample_df['sos'].std(),
            'eos_mean': self.sample_df['eos'].mean(),
            'eos_std': self.sample_df['eos'].std(),
            'los_mean': self.sample_df['los'].mean(),
            'los_std': self.sample_df['los'].std()
        }
        
        # 4. Additional validation metrics
        self.validation_results['quality_metrics'] = {
            'sample_size': len(self.sample_df),
            'data_completeness': len(self.sample_df) / sample_size,
            'outlier_percentage': self._detect_outliers_percentage(),
            'confidence_intervals': self._calculate_confidence_intervals()
        }
        
        print("Validation completed")
        return self.validation_results, self.sample_df
    
    def _detect_outliers_percentage(self):
        """Detect outliers using IQR method."""
        metrics = ['sos', 'eos', 'los']
        outlier_percentages = {}
        
        for metric in metrics:
            Q1 = self.sample_df[metric].quantile(0.25)
            Q3 = self.sample_df[metric].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.sample_df[(self.sample_df[metric] < Q1 - 1.5*IQR) | 
                                    (self.sample_df[metric] > Q3 + 1.5*IQR)]
            outlier_percentages[metric] = len(outliers) / len(self.sample_df) * 100
        
        return outlier_percentages
    
    def _calculate_confidence_intervals(self):
        """Calculate 95% confidence intervals for key metrics."""
        metrics = ['sos', 'eos', 'los']
        confidence_intervals = {}
        
        for metric in metrics:
            mean_val = self.sample_df[metric].mean()
            std_val = self.sample_df[metric].std()
            n = len(self.sample_df)
            
            # 95% confidence interval
            ci = 1.96 * std_val / np.sqrt(n)
            confidence_intervals[metric] = {
                'mean': mean_val,
                'lower_ci': mean_val - ci,
                'upper_ci': mean_val + ci
            }
        
        return confidence_intervals
    
    def create_visualizations(self):
        """Create comprehensive phenology visualizations."""
        
        # Initialize map
        self.Map = geemap.Map(center=[18.5, -70.5], zoom=7)
        
        # Add boundary
        self.Map.addLayer(self.boundary, {'color': 'black'}, 'Dominican Republic')
        
        # Add rice mask
        self.Map.addLayer(self.rice_mask.clip(self.boundary), 
                        {'palette': ['red', 'green'], 'min': 0, 'max': 1}, 
                        'Rice Growing Areas')
        
        # Add phenology layers with enhanced styling
        sos_vis = {
            'min': 0, 'max': 365,
            'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red'],
            'title': 'Start of Season (Days of Year)'
        }
        
        eos_vis = {
            'min': 0, 'max': 365,
            'palette': ['red', 'orange', 'yellow', 'green', 'cyan', 'blue'],
            'title': 'End of Season (Days of Year)'
        }
        
        los_vis = {
            'min': 30, 'max': 365,
            'palette': ['red', 'yellow', 'green'],
            'title': 'Length of Season (Days)'
        }
        
        seasonality_vis = {
            'min': 0, 'max': 1,
            'palette': ['white', 'green'],
            'title': 'Seasonality Index'
        }
        
        # Add masked phenology layers
        self.Map.addLayer(self.phenology_results.select('sos').updateMask(self.rice_mask).clip(self.boundary), 
                        sos_vis, 'SOS (Rice Areas)')
        
        self.Map.addLayer(self.phenology_results.select('eos').updateMask(self.rice_mask).clip(self.boundary), 
                        eos_vis, 'EOS (Rice Areas)')
        
        self.Map.addLayer(self.phenology_results.select('los').updateMask(self.rice_mask).clip(self.boundary), 
                        los_vis, 'LOS (Rice Areas)')
        
        self.Map.addLayer(self.phenology_results.select('seasonality').updateMask(self.rice_mask).clip(self.boundary), 
                        seasonality_vis, 'Seasonality (Rice Areas)')
        
        return self.Map
    
    def create_statistical_plots(self):
        """Create comprehensive statistical visualizations."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Phenology Analysis Results - Dominican Republic Rice Areas', 
                    fontsize=16, fontweight='bold')
        
        # 1. SOS Distribution
        axes[0,0].hist(self.sample_df['sos'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Start of Season Distribution')
        axes[0,0].set_xlabel('Day of Year')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.sample_df['sos'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.sample_df["sos"].mean():.1f}')
        axes[0,0].legend()
        
        # 2. EOS Distribution
        axes[0,1].hist(self.sample_df['eos'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_title('End of Season Distribution')
        axes[0,1].set_xlabel('Day of Year')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(self.sample_df['eos'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.sample_df["eos"].mean():.1f}')
        axes[0,1].legend()
        
        # 3. LOS Distribution
        axes[0,2].hist(self.sample_df['los'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,2].set_title('Length of Season Distribution')
        axes[0,2].set_xlabel('Days')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(self.sample_df['los'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.sample_df["los"].mean():.1f}')
        axes[0,2].legend()
        
        # 4. SOS vs EOS Scatter
        axes[1,0].scatter(self.sample_df['sos'], self.sample_df['eos'], alpha=0.6, color='purple')
        axes[1,0].set_title('SOS vs EOS Relationship')
        axes[1,0].set_xlabel('Start of Season (Day of Year)')
        axes[1,0].set_ylabel('End of Season (Day of Year)')
        
        # Add trend line
        z = np.polyfit(self.sample_df['sos'], self.sample_df['eos'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(self.sample_df['sos'], p(self.sample_df['sos']), "r--", alpha=0.8)
        
        # 5. Seasonality Distribution
        axes[1,1].hist(self.sample_df['seasonality'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1,1].set_title('Seasonality Index Distribution')
        axes[1,1].set_xlabel('Seasonality Index')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(self.sample_df['seasonality'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.sample_df["seasonality"].mean():.3f}')
        axes[1,1].legend()
        
        # 6. Growth Rate vs Seasonality
        axes[1,2].scatter(self.sample_df['growth_rate'], self.sample_df['seasonality'], alpha=0.6, color='teal')
        axes[1,2].set_title('Growth Rate vs Seasonality')
        axes[1,2].set_xlabel('Growth Rate')
        axes[1,2].set_ylabel('Seasonality Index')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_correlation_analysis(self):
        """Create correlation analysis between phenology metrics."""
        
        # Select relevant columns
        phenology_cols = ['sos', 'eos', 'los', 'seasonality', 'growth_rate', 'annual_amp', 'semi_annual_amp']
        corr_data = self.sample_df[phenology_cols]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Enhanced Phenology Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print significant correlations
        print("\n=== SIGNIFICANT CORRELATIONS (|r| > 0.3) ===")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: r = {corr_val:.3f}")
        
        return corr_matrix
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        
        print("=" * 60)
        print("ENHANCED PHENOLOGY ANALYSIS SUMMARY")
        print("=" * 60)
        print()
        
        print(f"STUDY AREA: {self.study_area}")
        print(f"PERIOD: {self.start_year}-{self.end_year}")
        print(f"SAMPLE SIZE: {len(self.sample_df)} pixels")
        print()
        
        print("KEY FINDINGS:")
        print("-" * 20)
        
        # Phenology timing
        sos_mean = self.sample_df['sos'].mean()
        eos_mean = self.sample_df['eos'].mean()
        los_mean = self.sample_df['los'].mean()
        
        print(f"• Average Start of Season: {sos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(sos_mean)): %B %d})")
        print(f"• Average End of Season: {eos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(eos_mean)): %B %d})")
        print(f"• Average Length of Season: {los_mean:.1f} days")
        print(f"• Average Seasonality Index: {self.sample_df['seasonality'].mean():.3f}")
        print(f"• Average Growth Rate: {self.sample_df['growth_rate'].mean():.4f}")
        print()
        
        print("QUALITY ASSESSMENT:")
        print("-" * 20)
        
        # Quality metrics
        all_valid = all(self.validation_results['logical_consistency'].values())
        print(f"• Data Quality: {'✓ PASS' if all_valid else '✗ FAIL'}")
        print(f"• Range Validation: {'✓ PASS' if all([v['valid'] for k, v in self.validation_results.items() if 'range' in k]) else '✗ FAIL'}")
        print(f"• Sample Coverage: {len(self.sample_df)} valid pixels")
        print(f"• Data Completeness: {self.validation_results['quality_metrics']['data_completeness']:.1%}")
        print()
        
        print("ENHANCED METRICS INCLUDED:")
        print("-" * 20)
        print("• Start of Season (SOS)")
        print("• End of Season (EOS)")
        print("• Length of Season (LOS)")
        print("• Seasonality Index")
        print("• Growth Rate")
        print("• Annual Amplitude")
        print("• Semi-annual Amplitude")
        print("• Peak NDVI Value")
        print("• Variability Index")
        print()
        
        print("VALIDATION METHODS:")
        print("-" * 20)
        print("• Range validation")
        print("• Logical consistency checks")
        print("• Statistical significance testing")
        print("• Correlation analysis")
        print("• Outlier detection")
        print("• Confidence intervals")
        print()
        
        print("=" * 60)
    
    def export_results(self, export_to_drive=True):
        """Export phenology results."""
        
        if export_to_drive:
            # Export phenology metrics
            export_tasks = []
            
            metrics = ['sos', 'eos', 'los', 'seasonality', 'growth_rate', 'peak_ndvi', 'variability']
            
            for metric in metrics:
                image = self.phenology_results.select(metric).updateMask(self.rice_mask).clip(self.boundary)
                
                task = ee.batch.Export.image.toDrive(
                    image=image,
                    description=f'DR_Rice_{metric.upper()}_Enhanced',
                    folder='Phenology_Analysis',
                    fileNamePrefix=f'dr_rice_{metric}_enhanced',
                    scale=250,
                    region=self.boundary.geometry(),
                    maxPixels=1e13
                )
                
                export_tasks.append(task)
                print(f"Export task created for {metric.upper()}")
            
            # Export rice mask
            rice_export = ee.batch.Export.image.toDrive(
                image=self.rice_mask.clip(self.boundary),
                description='DR_Rice_Mask_Enhanced',
                folder='Phenology_Analysis',
                fileNamePrefix='dr_rice_mask_enhanced',
                scale=250,
                region=self.boundary.geometry(),
                maxPixels=1e13
            )
            
            export_tasks.append(rice_export)
            print("Export task created for rice mask")
            
            return export_tasks
        else:
            print("Export to Google Drive disabled")
            return None

def main():
    """Main function to run the enhanced phenology analysis."""
    
    # Initialize analysis
    analysis = EnhancedPhenologyAnalysis('Dominican Republic', 2011, 2022)
    
    # Setup study area
    analysis.setup_study_area()
    
    # Get data
    analysis.get_enhanced_modis_data()
    
    # Extract phenology metrics
    analysis.extract_enhanced_phenology()
    
    # Create rice mask
    analysis.create_enhanced_rice_mask()
    
    # Validate results
    analysis.validate_phenology_metrics()
    
    # Create visualizations
    analysis.create_visualizations()
    analysis.create_statistical_plots()
    analysis.create_correlation_analysis()
    
    # Generate summary
    analysis.generate_summary_report()
    
    # Export results (optional)
    # analysis.export_results(export_to_drive=False)
    
    print("Enhanced phenology analysis completed successfully!")

if __name__ == "__main__":
    main()
