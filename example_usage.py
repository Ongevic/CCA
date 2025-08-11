#!/usr/bin/env python3
"""
Example Usage of Enhanced Phenology Analysis
============================================

This script demonstrates how to use the EnhancedPhenologyAnalysis class
for rice cultivation studies in the Dominican Republic.
"""

import sys
import os

# Add the current directory to the path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_phenology_analysis import EnhancedPhenologyAnalysis

def run_basic_analysis():
    """Run a basic phenology analysis."""
    
    print("🌱 Starting Enhanced Phenology Analysis")
    print("=" * 50)
    
    # Initialize analysis for Dominican Republic
    analysis = EnhancedPhenologyAnalysis('Dominican Republic', 2011, 2022)
    
    print("✅ Analysis initialized")
    
    # Setup study area
    boundary = analysis.setup_study_area()
    print("✅ Study area configured")
    
    # Get MODIS data
    modis_data = analysis.get_enhanced_modis_data()
    print("✅ MODIS data collected")
    
    # Extract phenology metrics
    phenology_results = analysis.extract_enhanced_phenology()
    print("✅ Phenology metrics extracted")
    
    # Create rice mask
    rice_mask = analysis.create_enhanced_rice_mask()
    print("✅ Rice mask created")
    
    # Validate results
    validation_results, sample_data = analysis.validate_phenology_metrics(sample_size=500)
    print("✅ Results validated")
    
    # Generate summary report
    analysis.generate_summary_report()
    
    print("\n🎉 Basic analysis completed successfully!")
    return analysis

def run_visualization_analysis():
    """Run analysis with visualizations."""
    
    print("\n📊 Creating Visualizations")
    print("=" * 50)
    
    # Get the analysis object from basic analysis
    analysis = run_basic_analysis()
    
    # Create interactive map
    print("Creating interactive map...")
    phenology_map = analysis.create_visualizations()
    print("✅ Interactive map created")
    
    # Create statistical plots
    print("Creating statistical plots...")
    stats_fig = analysis.create_statistical_plots()
    print("✅ Statistical plots created")
    
    # Create correlation analysis
    print("Creating correlation analysis...")
    corr_matrix = analysis.create_correlation_analysis()
    print("✅ Correlation analysis completed")
    
    print("\n🎉 Visualization analysis completed!")
    return analysis

def run_custom_analysis():
    """Run analysis with custom parameters."""
    
    print("\n🔧 Running Custom Analysis")
    print("=" * 50)
    
    # Custom parameters
    study_area = 'Dominican Republic'
    start_year = 2015
    end_year = 2020
    sample_size = 1000
    
    print(f"Study Area: {study_area}")
    print(f"Period: {start_year}-{end_year}")
    print(f"Sample Size: {sample_size}")
    
    # Initialize analysis
    analysis = EnhancedPhenologyAnalysis(study_area, start_year, end_year)
    
    # Run analysis
    analysis.setup_study_area()
    analysis.get_enhanced_modis_data()
    analysis.extract_enhanced_phenology()
    analysis.create_enhanced_rice_mask()
    analysis.validate_phenology_metrics(sample_size=sample_size)
    
    # Generate summary
    analysis.generate_summary_report()
    
    print("\n🎉 Custom analysis completed!")
    return analysis

def export_results_example():
    """Example of how to export results."""
    
    print("\n📤 Export Results Example")
    print("=" * 50)
    
    # Run basic analysis
    analysis = run_basic_analysis()
    
    # Export results (commented out to avoid accidental execution)
    print("Export functionality is available but disabled by default.")
    print("To enable export, uncomment the following line:")
    print("# export_tasks = analysis.export_results(export_to_drive=True)")
    
    # Example of what would be exported:
    print("\nFiles that would be exported:")
    print("- dr_rice_sos_enhanced.tif")
    print("- dr_rice_eos_enhanced.tif")
    print("- dr_rice_los_enhanced.tif")
    print("- dr_rice_seasonality_enhanced.tif")
    print("- dr_rice_growth_rate_enhanced.tif")
    print("- dr_rice_peak_ndvi_enhanced.tif")
    print("- dr_rice_variability_enhanced.tif")
    print("- dr_rice_mask_enhanced.tif")

def main():
    """Main function to run examples."""
    
    print("Enhanced Phenology Analysis - Example Usage")
    print("=" * 60)
    
    try:
        # Run basic analysis
        run_basic_analysis()
        
        # Run visualization analysis
        run_visualization_analysis()
        
        # Run custom analysis
        run_custom_analysis()
        
        # Show export example
        export_results_example()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated visualizations")
        print("2. Check the validation results")
        print("3. Modify parameters as needed")
        print("4. Export results if required")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Google Earth Engine is authenticated")
        print("2. Check internet connection")
        print("3. Verify all dependencies are installed")
        print("4. Check the study area and date range")

if __name__ == "__main__":
    main()
