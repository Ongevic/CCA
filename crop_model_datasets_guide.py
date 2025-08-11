#!/usr/bin/env python3
"""
Crop Model Calibration Datasets Guide
=====================================

Comprehensive guide for datasets that provide phenology and LAI data
for calibrating crop models like STICS for rice cultivation.

Author: Crop Model Dataset Guide
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CropModelDatasetGuide:
    """Guide for datasets useful in crop model calibration."""
    
    def __init__(self):
        """Initialize the dataset guide."""
        self.datasets = {
            'satellite': {},
            'ground_truth': {},
            'model_ready': {},
            'validation': {}
        }
        
    def get_satellite_datasets(self):
        """Get satellite-based datasets for phenology and LAI."""
        
        satellite_datasets = {
            'MODIS': {
                'name': 'MODIS (Moderate Resolution Imaging Spectroradiometer)',
                'products': {
                    'NDVI': {
                        'product': 'MODIS/061/MOD13Q1',
                        'resolution': '250m',
                        'frequency': '16-day',
                        'temporal_coverage': '2000-present',
                        'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak', 'Growth Rate'],
                        'advantages': ['Long time series', 'Global coverage', 'Free access'],
                        'limitations': ['Coarse resolution', 'Cloud contamination', 'Mixed pixels'],
                        'crop_model_use': 'Primary phenology extraction, seasonal patterns'
                    },
                    'LAI': {
                        'product': 'MODIS/061/MOD15A2H',
                        'resolution': '500m',
                        'frequency': '8-day',
                        'temporal_coverage': '2002-present',
                        'phenology_metrics': ['LAI time series', 'Peak LAI', 'LAI duration'],
                        'advantages': ['Direct LAI estimates', 'Long time series', 'Validated'],
                        'limitations': ['Coarse resolution', 'Saturation at high LAI', 'Uncertainty'],
                        'crop_model_use': 'LAI calibration, biomass estimation'
                    },
                    'EVI': {
                        'product': 'MODIS/061/MOD13Q1',
                        'resolution': '250m',
                        'frequency': '16-day',
                        'temporal_coverage': '2000-present',
                        'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak', 'Growth Rate'],
                        'advantages': ['Less saturation than NDVI', 'Better for dense vegetation'],
                        'limitations': ['Coarse resolution', 'Cloud contamination'],
                        'crop_model_use': 'Alternative to NDVI, dense crop monitoring'
                    }
                }
            },
            
            'Sentinel-2': {
                'name': 'Sentinel-2 MSI',
                'products': {
                    'NDVI': {
                        'product': 'COPERNICUS/S2_SR',
                        'resolution': '10m',
                        'frequency': '5-day',
                        'temporal_coverage': '2015-present',
                        'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak', 'Growth Rate'],
                        'advantages': ['High resolution', 'Frequent coverage', 'Good for small fields'],
                        'limitations': ['Shorter time series', 'Cloud contamination'],
                        'crop_model_use': 'High-resolution phenology, field-level analysis'
                    },
                    'LAI': {
                        'product': 'COPERNICUS/S2_SR (derived)',
                        'resolution': '10m',
                        'frequency': '5-day',
                        'temporal_coverage': '2015-present',
                        'phenology_metrics': ['LAI time series', 'Peak LAI', 'LAI duration'],
                        'advantages': ['High resolution', 'Can derive LAI from bands'],
                        'limitations': ['Requires LAI estimation algorithms', 'Validation needed'],
                        'crop_model_use': 'High-resolution LAI, field-level calibration'
                    }
                }
            },
            
            'Landsat': {
                'name': 'Landsat 8/9 OLI',
                'products': {
                    'NDVI': {
                        'product': 'LANDSAT/LC08/C02/T1_L2',
                        'resolution': '30m',
                        'frequency': '16-day',
                        'temporal_coverage': '2013-present',
                        'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak', 'Growth Rate'],
                        'advantages': ['Medium resolution', 'Long time series', 'Good for regional studies'],
                        'limitations': ['Less frequent', 'Cloud contamination'],
                        'crop_model_use': 'Regional phenology, historical analysis'
                    }
                }
            },
            
            'VIIRS': {
                'name': 'VIIRS (Visible Infrared Imaging Radiometer Suite)',
                'products': {
                    'NDVI': {
                        'product': 'NOAA/VIIRS/001/VNP13A1',
                        'resolution': '500m',
                        'frequency': '16-day',
                        'temporal_coverage': '2012-present',
                        'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak', 'Growth Rate'],
                        'advantages': ['Continuation of MODIS', 'Improved sensors'],
                        'limitations': ['Coarse resolution', 'Shorter time series'],
                        'crop_model_use': 'MODIS continuation, improved phenology'
                    }
                }
            }
        }
        
        return satellite_datasets
    
    def get_ground_truth_datasets(self):
        """Get ground truth datasets for validation and calibration."""
        
        ground_datasets = {
            'PhenoCam': {
                'name': 'PhenoCam Network',
                'description': 'Ground-based camera network for phenology monitoring',
                'coverage': 'Global (mainly US, Europe)',
                'temporal_coverage': '2008-present',
                'phenology_metrics': ['SOS', 'EOS', 'LOS', 'Peak timing', 'Greenness'],
                'advantages': ['High temporal resolution', 'Direct phenology observation', 'Validated'],
                'limitations': ['Limited spatial coverage', 'Point observations'],
                'crop_model_use': 'Validation of satellite phenology, local calibration',
                'access': 'https://phenocam.sr.unh.edu/'
            },
            
            'FLUXNET': {
                'name': 'FLUXNET Network',
                'description': 'Global network of eddy covariance flux towers',
                'coverage': 'Global',
                'temporal_coverage': '1990-present',
                'phenology_metrics': ['GPP phenology', 'LAI estimates', 'Carbon fluxes'],
                'advantages': ['Direct ecosystem measurements', 'High temporal resolution', 'Validated'],
                'limitations': ['Point observations', 'Limited crop sites'],
                'crop_model_use': 'Validation of LAI and phenology, carbon cycle calibration',
                'access': 'https://fluxnet.org/'
            },
            
            'LTER': {
                'name': 'Long Term Ecological Research Network',
                'description': 'Long-term ecological research sites',
                'coverage': 'US',
                'temporal_coverage': '1980-present',
                'phenology_metrics': ['Plant phenology', 'LAI measurements', 'Biomass'],
                'advantages': ['Long time series', 'Multiple variables', 'Well documented'],
                'limitations': ['US only', 'Limited crop sites'],
                'crop_model_use': 'Long-term validation, ecosystem dynamics',
                'access': 'https://lternet.edu/'
            },
            
            'NEON': {
                'name': 'National Ecological Observatory Network',
                'description': 'National network of ecological monitoring sites',
                'coverage': 'US',
                'temporal_coverage': '2013-present',
                'phenology_metrics': ['Plant phenology', 'LAI', 'Biomass', 'Canopy structure'],
                'advantages': ['Standardized protocols', 'Multiple variables', 'High quality'],
                'limitations': ['US only', 'Limited crop sites'],
                'crop_model_use': 'High-quality validation, multiple variables',
                'access': 'https://www.neonscience.org/'
            }
        }
        
        return ground_datasets
    
    def get_model_ready_datasets(self):
        """Get datasets specifically designed for crop model calibration."""
        
        model_datasets = {
            'GLAM': {
                'name': 'Global Land Analysis and Discovery (GLAD)',
                'description': 'Global agricultural monitoring system',
                'coverage': 'Global',
                'temporal_coverage': '2000-present',
                'phenology_metrics': ['Crop calendars', 'Growing seasons', 'Harvest dates'],
                'advantages': ['Crop-specific', 'Global coverage', 'Model ready'],
                'limitations': ['Coarse resolution', 'Limited variables'],
                'crop_model_use': 'Crop calendar initialization, growing season definition',
                'access': 'https://glad.umd.edu/'
            },
            
            'SPAM': {
                'name': 'Spatial Production Allocation Model',
                'description': 'Global crop production and area data',
                'coverage': 'Global',
                'temporal_coverage': '2000, 2005, 2010',
                'phenology_metrics': ['Crop areas', 'Production', 'Yield'],
                'advantages': ['Crop-specific', 'High resolution', 'Validated'],
                'limitations': ['Limited years', 'No temporal dynamics'],
                'crop_model_use': 'Crop area masks, yield validation',
                'access': 'https://www.mapspam.info/'
            },
            
            'MIRCA2000': {
                'name': 'Monthly Irrigated and Rainfed Crop Areas',
                'description': 'Global crop calendar and irrigation data',
                'coverage': 'Global',
                'temporal_coverage': '2000',
                'phenology_metrics': ['Crop calendars', 'Growing seasons', 'Irrigation'],
                'advantages': ['Crop-specific calendars', 'Irrigation information', 'Model ready'],
                'limitations': ['Single year', 'Coarse resolution'],
                'crop_model_use': 'Crop calendar initialization, irrigation modeling',
                'access': 'https://www.uni-frankfurt.de/45218023/MIRCA'
            },
            
            'GEOGLAM': {
                'name': 'Group on Earth Observations Global Agricultural Monitoring',
                'description': 'Global agricultural monitoring initiative',
                'coverage': 'Global',
                'temporal_coverage': '2010-present',
                'phenology_metrics': ['Crop conditions', 'Yield forecasts', 'Phenology'],
                'advantages': ['Crop-specific', 'Operational', 'Multiple sources'],
                'limitations': ['Limited public access', 'Coarse resolution'],
                'crop_model_use': 'Crop condition monitoring, yield validation',
                'access': 'https://cropmonitor.org/'
            }
        }
        
        return model_datasets
    
    def get_validation_datasets(self):
        """Get datasets for validating crop model outputs."""
        
        validation_datasets = {
            'FAOSTAT': {
                'name': 'FAO Statistical Database',
                'description': 'Global agricultural statistics',
                'coverage': 'Global',
                'temporal_coverage': '1961-present',
                'phenology_metrics': ['Production', 'Area', 'Yield', 'Harvest dates'],
                'advantages': ['Official statistics', 'Long time series', 'Global coverage'],
                'limitations': ['National level', 'Limited spatial detail'],
                'crop_model_use': 'Yield validation, production estimates',
                'access': 'http://www.fao.org/faostat/'
            },
            
            'USDA_NASS': {
                'name': 'USDA National Agricultural Statistics Service',
                'description': 'US agricultural statistics',
                'coverage': 'US',
                'temporal_coverage': '1866-present',
                'phenology_metrics': ['Production', 'Area', 'Yield', 'Progress reports'],
                'advantages': ['High quality', 'Detailed', 'Frequent updates'],
                'limitations': ['US only', 'Limited variables'],
                'crop_model_use': 'US crop validation, detailed statistics',
                'access': 'https://www.nass.usda.gov/'
            },
            
            'CropScape': {
                'name': 'USDA Cropland Data Layer',
                'description': 'Annual crop type classification',
                'coverage': 'US',
                'temporal_coverage': '2008-present',
                'phenology_metrics': ['Crop types', 'Crop areas', 'Crop rotations'],
                'advantages': ['High resolution', 'Annual updates', 'Validated'],
                'limitations': ['US only', 'No temporal dynamics'],
                'crop_model_use': 'Crop type validation, area estimation',
                'access': 'https://nassgeodata.gmu.edu/CropScape/'
            }
        }
        
        return validation_datasets
    
    def get_stics_specific_datasets(self):
        """Get datasets specifically useful for STICS model calibration."""
        
        stics_datasets = {
            'phenology': {
                'primary': 'MODIS NDVI (MOD13Q1)',
                'secondary': 'Sentinel-2 NDVI',
                'ground_truth': 'PhenoCam, FLUXNET',
                'use_in_stics': 'Initialize sowing dates, growing season length',
                'stics_parameters': ['date_debut', 'date_fin', 'durvieF']
            },
            
            'lai': {
                'primary': 'MODIS LAI (MOD15A2H)',
                'secondary': 'Sentinel-2 derived LAI',
                'ground_truth': 'FLUXNET, NEON',
                'use_in_stics': 'Calibrate LAI development, biomass allocation',
                'stics_parameters': ['sla', 'tigefeuil', 'rapforme']
            },
            
            'biomass': {
                'primary': 'MODIS NPP (MOD17A3H)',
                'secondary': 'Sentinel-2 derived biomass',
                'ground_truth': 'FLUXNET, field measurements',
                'use_in_stics': 'Calibrate biomass production, yield estimation',
                'stics_parameters': ['ebmax', 'topt', 'tmin']
            },
            
            'soil': {
                'primary': 'SoilGrids250m',
                'secondary': 'Harmonized World Soil Database',
                'ground_truth': 'Local soil surveys',
                'use_in_stics': 'Initialize soil parameters, water balance',
                'stics_parameters': ['profhum', 'calc', 'argi', 'Norg']
            },
            
            'climate': {
                'primary': 'ERA5, MERRA-2',
                'secondary': 'CHIRPS, PERSIANN',
                'ground_truth': 'Weather stations',
                'use_in_stics': 'Climate forcing, water balance',
                'stics_parameters': ['tmin', 'tmax', 'precip', 'solar_rad']
            }
        }
        
        return stics_datasets
    
    def create_dataset_comparison(self):
        """Create a comparison table of datasets."""
        
        print("🌾 CROP MODEL CALIBRATION DATASETS COMPARISON")
        print("=" * 80)
        
        # Satellite datasets
        print("\n📡 SATELLITE DATASETS FOR PHENOLOGY & LAI:")
        print("-" * 50)
        
        satellite_data = self.get_satellite_datasets()
        for platform, info in satellite_data.items():
            print(f"\n🔸 {platform.upper()}:")
            for product, details in info['products'].items():
                print(f"  • {product}: {details['resolution']}, {details['frequency']}")
                print(f"    Phenology: {', '.join(details['phenology_metrics'])}")
                print(f"    Crop Model Use: {details['crop_model_use']}")
        
        # Ground truth datasets
        print("\n🌱 GROUND TRUTH DATASETS FOR VALIDATION:")
        print("-" * 50)
        
        ground_data = self.get_ground_truth_datasets()
        for network, info in ground_data.items():
            print(f"\n🔸 {network.upper()}:")
            print(f"  • Coverage: {info['coverage']}")
            print(f"  • Phenology: {', '.join(info['phenology_metrics'])}")
            print(f"  • Crop Model Use: {info['crop_model_use']}")
        
        # Model-ready datasets
        print("\n🎯 MODEL-READY DATASETS:")
        print("-" * 50)
        
        model_data = self.get_model_ready_datasets()
        for dataset, info in model_data.items():
            print(f"\n🔸 {dataset.upper()}:")
            print(f"  • Coverage: {info['coverage']}")
            print(f"  • Phenology: {', '.join(info['phenology_metrics'])}")
            print(f"  • Crop Model Use: {info['crop_model_use']}")
        
        # STICS-specific recommendations
        print("\n🌾 STICS MODEL SPECIFIC RECOMMENDATIONS:")
        print("-" * 50)
        
        stics_data = self.get_stics_specific_datasets()
        for variable, info in stics_data.items():
            print(f"\n🔸 {variable.upper()}:")
            print(f"  • Primary: {info['primary']}")
            print(f"  • Secondary: {info['secondary']}")
            print(f"  • Ground Truth: {info['ground_truth']}")
            print(f"  • STICS Use: {info['use_in_stics']}")
            print(f"  • STICS Parameters: {', '.join(info['stics_parameters'])}")
    
    def create_implementation_guide(self):
        """Create implementation guide for using these datasets."""
        
        print("\n🚀 IMPLEMENTATION GUIDE FOR CROP MODEL CALIBRATION")
        print("=" * 80)
        
        print("\n📋 STEP-BY-STEP APPROACH:")
        print("1. PHENOLOGY EXTRACTION:")
        print("   • Use MODIS NDVI (MOD13Q1) for primary phenology extraction")
        print("   • Apply harmonic analysis to extract SOS, EOS, LOS")
        print("   • Validate with PhenoCam or FLUXNET ground data")
        print("   • Use for STICS sowing date initialization")
        
        print("\n2. LAI ESTIMATION:")
        print("   • Use MODIS LAI (MOD15A2H) for primary LAI data")
        print("   • Apply temporal smoothing and gap filling")
        print("   • Validate with ground LAI measurements")
        print("   • Use for STICS LAI development calibration")
        
        print("\n3. CROP MODEL CALIBRATION:")
        print("   • Initialize STICS with extracted phenology dates")
        print("   • Calibrate LAI parameters using satellite LAI")
        print("   • Validate biomass/yield with ground measurements")
        print("   • Iterate calibration using multiple years")
        
        print("\n4. VALIDATION:")
        print("   • Compare model outputs with ground truth data")
        print("   • Use multiple validation datasets (FAOSTAT, USDA)")
        print("   • Assess model performance across different years")
        print("   • Document uncertainties and limitations")
        
        print("\n🎯 RECOMMENDED DATASET COMBINATION FOR RICE:")
        print("• Primary Phenology: MODIS NDVI (MOD13Q1)")
        print("• Primary LAI: MODIS LAI (MOD15A2H)")
        print("• Ground Validation: PhenoCam, FLUXNET")
        print("• Crop Calendar: MIRCA2000, GLAM")
        print("• Yield Validation: FAOSTAT, local statistics")
        
        print("\n⚠️ IMPORTANT CONSIDERATIONS:")
        print("• Cloud contamination affects satellite data quality")
        print("• Ground truth data may be limited for rice")
        print("• Temporal resolution varies between datasets")
        print("• Spatial resolution affects field-level analysis")
        print("• Data access may require registration/approval")

def main():
    """Main function to run the dataset guide."""
    
    print("🌾 Crop Model Calibration Datasets Guide")
    print("=" * 80)
    
    guide = CropModelDatasetGuide()
    
    # Create comprehensive comparison
    guide.create_dataset_comparison()
    
    # Create implementation guide
    guide.create_implementation_guide()
    
    print("\n✅ Dataset guide completed!")
    print("\n📚 Next Steps:")
    print("1. Choose appropriate datasets for your study area")
    print("2. Set up data access and download procedures")
    print("3. Implement data processing and quality control")
    print("4. Integrate with your crop model calibration workflow")

if __name__ == "__main__":
    main()

