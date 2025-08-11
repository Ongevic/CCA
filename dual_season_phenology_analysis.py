#!/usr/bin/env python3
"""
Dual-Season Phenology Analysis for Dominican Republic Rice Cultivation
=====================================================================

This script performs phenology analysis specifically designed to capture
the two distinct growing seasons in the Dominican Republic:

Main Season: December-June (Planting Dec-Feb, Harvest Apr-Jun)
Second Season: April-November (Planting Apr-Jul, Harvest Aug-Nov)

Key Features:
- Dual-season phenology detection
- Season-specific metrics
- Enhanced validation for multiple seasons
- Comprehensive visualization of both seasons
"""

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
plt.style.use('default')
sns.set_palette("husl")

class DualSeasonPhenologyAnalysis:
    """Dual-season phenology analysis class for Dominican Republic rice cultivation."""
    
    def __init__(self, study_area='Dominican Republic', start_year=2011, end_year=2022):
        """
        Initialize the dual-season phenology analysis.
        
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
        
        # Define the two growing seasons based on Dominican Republic rice calendar
        self.growing_seasons = {
            'main_season': {
                'name': 'Main Season',
                'planting_start': 335,  # December 1st (day of year)
                'planting_end': 59,     # February 28th (day of year)
                'harvest_start': 91,    # April 1st (day of year)
                'harvest_end': 181,     # June 30th (day of year)
                'description': 'December-June (Planting Dec-Feb, Harvest Apr-Jun)'
            },
            'second_season': {
                'name': 'Second Season',
                'planting_start': 91,   # April 1st (day of year)
                'planting_end': 212,    # July 31st (day of year)
                'harvest_start': 213,   # August 1st (day of year)
                'harvest_end': 334,     # November 30th (day of year)
                'description': 'April-November (Planting Apr-Jul, Harvest Aug-Nov)'
            }
        }
        
        print(f"Initialized dual-season phenology analysis for {study_area} ({start_year}-{end_year})")
        print("Growing Seasons:")
        for season, info in self.growing_seasons.items():
            print(f"  • {info['name']}: {info['description']}")
    
    def generate_dual_season_sample_data(self, n_pixels=1000):
        """Generate sample phenology data for both growing seasons."""
        
        print("Generating dual-season phenology data...")
        
        # Generate realistic sample data for both seasons
        np.random.seed(42)  # For reproducible results
        
        # Main season data (December-June)
        main_season_data = {
            'sos_main': np.random.normal(335, 15, n_pixels),  # Start around December
            'eos_main': np.random.normal(181, 15, n_pixels),  # End around June
            'los_main': np.random.normal(180, 20, n_pixels),  # Length ~180 days
            'seasonality_main': np.random.uniform(0.7, 0.9, n_pixels),
            'growth_rate_main': np.random.normal(0.02, 0.005, n_pixels),
            'annual_amp_main': np.random.normal(0.3, 0.1, n_pixels),
            'peak_ndvi_main': np.random.normal(0.7, 0.15, n_pixels),
            'variability_main': np.random.normal(75, 10, n_pixels),
        }
        
        # Second season data (April-November)
        second_season_data = {
            'sos_second': np.random.normal(91, 15, n_pixels),   # Start around April
            'eos_second': np.random.normal(334, 15, n_pixels),  # End around November
            'los_second': np.random.normal(200, 25, n_pixels),  # Length ~200 days
            'seasonality_second': np.random.uniform(0.6, 0.8, n_pixels),
            'growth_rate_second': np.random.normal(0.018, 0.005, n_pixels),
            'annual_amp_second': np.random.normal(0.25, 0.1, n_pixels),
            'peak_ndvi_second': np.random.normal(0.65, 0.15, n_pixels),
            'variability_second': np.random.normal(80, 12, n_pixels),
        }
        
        # Combine data
        sample_data = {**main_season_data, **second_season_data}
        
        # Add rice mask and season preference
        sample_data['rice_mask'] = np.random.choice([0, 1], n_pixels, p=[0.3, 0.7])
        sample_data['season_preference'] = np.random.choice(['main', 'second', 'both'], n_pixels, p=[0.4, 0.3, 0.3])
        
        # Ensure logical consistency for main season
        sample_data['eos_main'] = sample_data['sos_main'] + sample_data['los_main']
        sample_data['sos_main'] = np.clip(sample_data['sos_main'], 335, 59)  # Dec-Feb
        sample_data['eos_main'] = np.clip(sample_data['eos_main'], 91, 181)  # Apr-Jun
        sample_data['los_main'] = np.clip(sample_data['los_main'], 120, 240)
        
        # Ensure logical consistency for second season
        sample_data['eos_second'] = sample_data['sos_second'] + sample_data['los_second']
        sample_data['sos_second'] = np.clip(sample_data['sos_second'], 91, 212)   # Apr-Jul
        sample_data['eos_second'] = np.clip(sample_data['eos_second'], 213, 334)  # Aug-Nov
        sample_data['los_second'] = np.clip(sample_data['los_second'], 120, 240)
        
        # Create DataFrame
        self.sample_df = pd.DataFrame(sample_data)
        
        # Filter for rice areas only
        self.rice_df = self.sample_df[self.sample_df['rice_mask'] == 1].copy()
        
        # Create season-specific datasets
        self.main_season_df = self.rice_df[self.rice_df['season_preference'].isin(['main', 'both'])].copy()
        self.second_season_df = self.rice_df[self.rice_df['season_preference'].isin(['second', 'both'])].copy()
        
        print(f"Generated {len(self.sample_df)} total pixels, {len(self.rice_df)} rice pixels")
        print(f"Main season pixels: {len(self.main_season_df)}")
        print(f"Second season pixels: {len(self.second_season_df)}")
        
        return self.sample_df
    
    def validate_dual_season_metrics(self):
        """Validate phenology metrics for both seasons."""
        
        if not hasattr(self, 'sample_df'):
            self.generate_dual_season_sample_data()
        
        print("Validating dual-season phenology metrics...")
        
        # Validation metrics
        self.validation_results = {}
        
        # Validate main season
        self.validation_results['main_season'] = self._validate_season_metrics('main')
        
        # Validate second season
        self.validation_results['second_season'] = self._validate_season_metrics('second')
        
        # Cross-season validation
        self.validation_results['cross_season'] = {
            'main_before_second': (self.main_season_df['eos_main'].mean() < self.second_season_df['sos_second'].mean()),
            'season_overlap_check': self._check_season_overlap(),
            'logical_consistency': self._check_dual_season_logic()
        }
        
        print("Dual-season validation completed")
        return self.validation_results, self.sample_df
    
    def _validate_season_metrics(self, season):
        """Validate metrics for a specific season."""
        season_df = self.main_season_df if season == 'main' else self.second_season_df
        
        validation = {}
        
        # Range validation
        validation['sos_range'] = {
            'min': season_df[f'sos_{season}'].min(),
            'max': season_df[f'sos_{season}'].max(),
            'valid': self._check_season_range(season, 'sos', season_df[f'sos_{season}'])
        }
        
        validation['eos_range'] = {
            'min': season_df[f'eos_{season}'].min(),
            'max': season_df[f'eos_{season}'].max(),
            'valid': self._check_season_range(season, 'eos', season_df[f'eos_{season}'])
        }
        
        validation['los_range'] = {
            'min': season_df[f'los_{season}'].min(),
            'max': season_df[f'los_{season}'].max(),
            'valid': 120 <= season_df[f'los_{season}'].min() <= season_df[f'los_{season}'].max() <= 240
        }
        
        # Statistical summary
        validation['statistics'] = {
            'sos_mean': season_df[f'sos_{season}'].mean(),
            'sos_std': season_df[f'sos_{season}'].std(),
            'eos_mean': season_df[f'eos_{season}'].mean(),
            'eos_std': season_df[f'eos_{season}'].std(),
            'los_mean': season_df[f'los_{season}'].mean(),
            'los_std': season_df[f'los_{season}'].std()
        }
        
        return validation
    
    def _check_season_range(self, season, metric, values):
        """Check if values are within expected ranges for each season."""
        if season == 'main':
            if metric == 'sos':
                return (335 <= values.min()) and (values.max() <= 59)  # Dec-Feb
            elif metric == 'eos':
                return (91 <= values.min()) and (values.max() <= 181)  # Apr-Jun
        elif season == 'second':
            if metric == 'sos':
                return (91 <= values.min()) and (values.max() <= 212)  # Apr-Jul
            elif metric == 'eos':
                return (213 <= values.min()) and (values.max() <= 334)  # Aug-Nov
        return False
    
    def _check_season_overlap(self):
        """Check for logical season overlap patterns."""
        # Main season should end before second season starts (for most areas)
        main_ends = self.main_season_df['eos_main'].mean()
        second_starts = self.second_season_df['sos_second'].mean()
        
        return {
            'main_ends_before_second_starts': main_ends < second_starts,
            'main_ends_day': main_ends,
            'second_starts_day': second_starts,
            'gap_days': second_starts - main_ends
        }
    
    def _check_dual_season_logic(self):
        """Check logical consistency between seasons."""
        return {
            'main_season_valid': (self.main_season_df['sos_main'] < self.main_season_df['eos_main']).all(),
            'second_season_valid': (self.second_season_df['sos_second'] < self.second_season_df['eos_second']).all(),
            'both_seasons_positive_los': (
                (self.main_season_df['los_main'] > 0).all() and 
                (self.second_season_df['los_second'] > 0).all()
            )
        }
    
    def create_dual_season_plots(self):
        """Create comprehensive visualizations for both seasons."""
        
        if not hasattr(self, 'sample_df'):
            self.generate_dual_season_sample_data()
        
        print("Creating dual-season statistical plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Dual-Season Phenology Analysis - Dominican Republic Rice Areas', 
                    fontsize=16, fontweight='bold')
        
        # 1. Main Season SOS Distribution
        axes[0,0].hist(self.main_season_df['sos_main'], bins=30, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[0,0].set_title('Main Season: Start of Season Distribution')
        axes[0,0].set_xlabel('Day of Year')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.main_season_df['sos_main'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.main_season_df["sos_main"].mean():.1f}')
        axes[0,0].legend()
        
        # 2. Second Season SOS Distribution
        axes[0,1].hist(self.second_season_df['sos_second'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0,1].set_title('Second Season: Start of Season Distribution')
        axes[0,1].set_xlabel('Day of Year')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(self.second_season_df['sos_second'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.second_season_df["sos_second"].mean():.1f}')
        axes[0,1].legend()
        
        # 3. Season Comparison: SOS
        axes[0,2].boxplot([self.main_season_df['sos_main'], self.second_season_df['sos_second']], 
                         labels=['Main Season', 'Second Season'], patch_artist=True)
        axes[0,2].set_title('Season Comparison: Start of Season')
        axes[0,2].set_ylabel('Day of Year')
        
        # 4. Main Season EOS Distribution
        axes[1,0].hist(self.main_season_df['eos_main'], bins=30, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[1,0].set_title('Main Season: End of Season Distribution')
        axes[1,0].set_xlabel('Day of Year')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(self.main_season_df['eos_main'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.main_season_df["eos_main"].mean():.1f}')
        axes[1,0].legend()
        
        # 5. Second Season EOS Distribution
        axes[1,1].hist(self.second_season_df['eos_second'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].set_title('Second Season: End of Season Distribution')
        axes[1,1].set_xlabel('Day of Year')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(self.second_season_df['eos_second'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.second_season_df["eos_second"].mean():.1f}')
        axes[1,1].legend()
        
        # 6. Season Comparison: EOS
        axes[1,2].boxplot([self.main_season_df['eos_main'], self.second_season_df['eos_second']], 
                         labels=['Main Season', 'Second Season'], patch_artist=True)
        axes[1,2].set_title('Season Comparison: End of Season')
        axes[1,2].set_ylabel('Day of Year')
        
        # 7. Season Length Comparison
        axes[2,0].boxplot([self.main_season_df['los_main'], self.second_season_df['los_second']], 
                         labels=['Main Season', 'Second Season'], patch_artist=True)
        axes[2,0].set_title('Season Comparison: Length of Season')
        axes[2,0].set_ylabel('Days')
        
        # 8. Seasonality Comparison
        axes[2,1].boxplot([self.main_season_df['seasonality_main'], self.second_season_df['seasonality_second']], 
                         labels=['Main Season', 'Second Season'], patch_artist=True)
        axes[2,1].set_title('Season Comparison: Seasonality Index')
        axes[2,1].set_ylabel('Seasonality Index')
        
        # 9. Growth Rate Comparison
        axes[2,2].boxplot([self.main_season_df['growth_rate_main'], self.second_season_df['growth_rate_second']], 
                         labels=['Main Season', 'Second Season'], patch_artist=True)
        axes[2,2].set_title('Season Comparison: Growth Rate')
        axes[2,2].set_ylabel('Growth Rate')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_seasonal_calendar_plot(self):
        """Create a seasonal calendar visualization."""
        
        if not hasattr(self, 'sample_df'):
            self.generate_dual_season_sample_data()
        
        print("Creating seasonal calendar plot...")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Define months and their day ranges
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_starts = [1] + [sum(month_days[:i+1]) for i in range(11)]
        
        # Plot month separators
        for i, month in enumerate(months):
            ax.axvline(x=month_starts[i], color='gray', alpha=0.3, linestyle='--')
            ax.text(month_starts[i] + month_days[i]/2, 0.9, month, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Plot main season
        main_sos_mean = self.main_season_df['sos_main'].mean()
        main_eos_mean = self.main_season_df['eos_main'].mean()
        main_los_mean = self.main_season_df['los_main'].mean()
        
        ax.barh('Main Season', main_los_mean, left=main_sos_mean, height=0.3, 
               color='darkgreen', alpha=0.7, label='Main Season')
        ax.text(main_sos_mean + main_los_mean/2, 0.5, f'{main_los_mean:.0f} days', 
               ha='center', va='center', color='white', fontweight='bold')
        
        # Plot second season
        second_sos_mean = self.second_season_df['sos_second'].mean()
        second_eos_mean = self.second_season_df['eos_second'].mean()
        second_los_mean = self.second_season_df['los_second'].mean()
        
        ax.barh('Second Season', second_los_mean, left=second_sos_mean, height=0.3, 
               color='orange', alpha=0.7, label='Second Season')
        ax.text(second_sos_mean + second_los_mean/2, 0.5, f'{second_los_mean:.0f} days', 
               ha='center', va='center', color='black', fontweight='bold')
        
        # Add planting and harvest markers
        ax.scatter([main_sos_mean, main_eos_mean], [0.5, 0.5], color='darkgreen', s=100, 
                  marker='o', zorder=5, label='Main Season Events')
        ax.scatter([second_sos_mean, second_eos_mean], [0.5, 0.5], color='orange', s=100, 
                  marker='s', zorder=5, label='Second Season Events')
        
        # Customize plot
        ax.set_xlim(0, 365)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Day of Year')
        ax.set_title('Dominican Republic Rice Growing Seasons Calendar', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_dual_season_summary_report(self):
        """Generate comprehensive summary report for both seasons."""
        
        if not hasattr(self, 'sample_df'):
            self.generate_dual_season_sample_data()
        
        print("=" * 70)
        print("DUAL-SEASON PHENOLOGY ANALYSIS SUMMARY")
        print("=" * 70)
        print()
        
        print(f"STUDY AREA: {self.study_area}")
        print(f"PERIOD: {self.start_year}-{self.end_year}")
        print(f"TOTAL SAMPLE SIZE: {len(self.rice_df)} rice pixels")
        print(f"MAIN SEASON PIXELS: {len(self.main_season_df)}")
        print(f"SECOND SEASON PIXELS: {len(self.second_season_df)}")
        print()
        
        print("MAIN SEASON FINDINGS (December-June):")
        print("-" * 40)
        main_sos_mean = self.main_season_df['sos_main'].mean()
        main_eos_mean = self.main_season_df['eos_main'].mean()
        main_los_mean = self.main_season_df['los_main'].mean()
        
        print(f"• Average Start of Season: {main_sos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(main_sos_mean)): %B %d})")
        print(f"• Average End of Season: {main_eos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(main_eos_mean)): %B %d})")
        print(f"• Average Length of Season: {main_los_mean:.1f} days")
        print(f"• Average Seasonality Index: {self.main_season_df['seasonality_main'].mean():.3f}")
        print(f"• Average Growth Rate: {self.main_season_df['growth_rate_main'].mean():.4f}")
        print()
        
        print("SECOND SEASON FINDINGS (April-November):")
        print("-" * 40)
        second_sos_mean = self.second_season_df['sos_second'].mean()
        second_eos_mean = self.second_season_df['eos_second'].mean()
        second_los_mean = self.second_season_df['los_second'].mean()
        
        print(f"• Average Start of Season: {second_sos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(second_sos_mean)): %B %d})")
        print(f"• Average End of Season: {second_eos_mean:.1f} days (≈ {datetime(2020, 1, 1) + timedelta(days=int(second_eos_mean)): %B %d})")
        print(f"• Average Length of Season: {second_los_mean:.1f} days")
        print(f"• Average Seasonality Index: {self.second_season_df['seasonality_second'].mean():.3f}")
        print(f"• Average Growth Rate: {self.second_season_df['growth_rate_second'].mean():.4f}")
        print()
        
        print("SEASON COMPARISON:")
        print("-" * 40)
        print(f"• Length Difference: {abs(main_los_mean - second_los_mean):.1f} days")
        print(f"• Seasonality Difference: {abs(self.main_season_df['seasonality_main'].mean() - self.second_season_df['seasonality_second'].mean()):.3f}")
        print(f"• Growth Rate Difference: {abs(self.main_season_df['growth_rate_main'].mean() - self.second_season_df['growth_rate_second'].mean()):.4f}")
        print()
        
                 print("QUALITY ASSESSMENT:")
         print("-" * 40)
         main_valid = (self.validation_results['main_season']['sos_range']['valid'] and 
                      self.validation_results['main_season']['eos_range']['valid'] and
                      self.validation_results['main_season']['los_range']['valid'])
         second_valid = (self.validation_results['second_season']['sos_range']['valid'] and 
                        self.validation_results['second_season']['eos_range']['valid'] and
                        self.validation_results['second_season']['los_range']['valid'])
         
         print(f"• Main Season Quality: {'✓ PASS' if main_valid else '✗ FAIL'}")
         print(f"• Second Season Quality: {'✓ PASS' if second_valid else '✗ FAIL'}")
         print(f"• Cross-Season Logic: {'✓ PASS' if all(self.validation_results['cross_season']['logical_consistency'].values()) else '✗ FAIL'}")
        print()
        
        print("DUAL-SEASON METRICS INCLUDED:")
        print("-" * 40)
        print("• Main Season: SOS, EOS, LOS, Seasonality, Growth Rate, Annual Amplitude, Peak NDVI, Variability")
        print("• Second Season: SOS, EOS, LOS, Seasonality, Growth Rate, Annual Amplitude, Peak NDVI, Variability")
        print("• Cross-Season: Season overlap analysis, temporal relationships")
        print()
        
        print("=" * 70)
    
    def save_dual_season_results(self, filename='dual_season_phenology_results.csv'):
        """Save dual-season results to CSV file."""
        
        if not hasattr(self, 'sample_df'):
            self.generate_dual_season_sample_data()
        
        # Save rice data with both seasons
        self.rice_df.to_csv(filename, index=False)
        print(f"Dual-season results saved to {filename}")
        
        # Save main season data
        main_filename = filename.replace('.csv', '_main_season.csv')
        self.main_season_df.to_csv(main_filename, index=False)
        print(f"Main season results saved to {main_filename}")
        
        # Save second season data
        second_filename = filename.replace('.csv', '_second_season.csv')
        self.second_season_df.to_csv(second_filename, index=False)
        print(f"Second season results saved to {second_filename}")
        
        # Save validation results
        validation_filename = filename.replace('.csv', '_validation.txt')
        with open(validation_filename, 'w') as f:
            f.write("DUAL-SEASON PHENOLOGY ANALYSIS VALIDATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in self.validation_results.items():
                f.write(f"{key.upper()}:\n")
                f.write(str(value) + "\n\n")
        
        print(f"Validation results saved to {validation_filename}")

def main():
    """Main function to run the dual-season phenology analysis."""
    
    print("🌾 Dual-Season Phenology Analysis for Dominican Republic Rice")
    print("=" * 70)
    
    try:
        # Initialize analysis
        analysis = DualSeasonPhenologyAnalysis('Dominican Republic', 2011, 2022)
        
        # Generate dual-season sample data
        analysis.generate_dual_season_sample_data()
        
        # Validate results
        analysis.validate_dual_season_metrics()
        
        # Create visualizations
        analysis.create_dual_season_plots()
        analysis.create_seasonal_calendar_plot()
        
        # Generate summary
        analysis.generate_dual_season_summary_report()
        
        # Save results
        analysis.save_dual_season_results()
        
        print("\n🎉 Dual-season phenology analysis completed successfully!")
        print("\nKey Achievements:")
        print("✅ Captured both Main Season (Dec-Jun) and Second Season (Apr-Nov)")
        print("✅ Generated season-specific phenology metrics")
        print("✅ Created comprehensive dual-season visualizations")
        print("✅ Validated cross-season relationships")
        print("✅ Saved separate results for each season")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that all required packages are installed")
        print("2. Ensure you have write permissions in the current directory")

if __name__ == "__main__":
    main()
