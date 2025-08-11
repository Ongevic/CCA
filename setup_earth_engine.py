#!/usr/bin/env python3
"""
Google Earth Engine Setup Guide
==============================

This script helps you set up Google Earth Engine access for the phenology analysis.
"""

import os
import sys

def print_setup_instructions():
    """Print step-by-step setup instructions."""
    
    print("🌍 GOOGLE EARTH ENGINE SETUP GUIDE")
    print("=" * 50)
    print()
    
    print("STEP 1: Sign up for Google Earth Engine")
    print("-" * 40)
    print("1. Visit: https://signup.earthengine.google.com/")
    print("2. Sign in with your Google account")
    print("3. Fill out the application form")
    print("4. Submit and wait for approval (1-2 days)")
    print()
    
    print("STEP 2: Install Required Packages")
    print("-" * 40)
    print("Run these commands in your terminal:")
    print("pip install earthengine-api geemap")
    print()
    
    print("STEP 3: Authenticate (After Approval)")
    print("-" * 40)
    print("Once approved, run this Python code:")
    print()
    print("import ee")
    print("ee.Authenticate()")
    print("ee.Initialize()")
    print()
    
    print("STEP 4: Test Installation")
    print("-" * 40)
    print("Run this test code:")
    print()
    print("import ee")
    print("print('Earth Engine version:', ee.__version__)")
    print("print('Authentication successful!')")
    print()
    
    print("STEP 5: Run Full Analysis")
    print("-" * 40)
    print("After setup, you can run the full analysis:")
    print("python enhanced_phenology_analysis.py")
    print()
    
    print("TROUBLESHOOTING")
    print("-" * 40)
    print("• If you get authentication errors, try:")
    print("  ee.Authenticate()")
    print("  ee.Initialize()")
    print()
    print("• If you get import errors, install packages:")
    print("  pip install earthengine-api geemap")
    print()
    print("• For more help, visit:")
    print("  https://developers.google.com/earth-engine/guides/access")
    print()

def test_earth_engine():
    """Test if Earth Engine is properly set up."""
    
    print("🧪 Testing Google Earth Engine Setup")
    print("=" * 40)
    
    try:
        import ee
        print("✅ Earth Engine imported successfully")
        print(f"   Version: {ee.__version__}")
        
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized successfully")
            
            # Test a simple operation
            test_image = ee.Image('USGS/SRTMGL1_003')
            print("✅ Earth Engine operations working")
            
            print("\n🎉 Earth Engine is ready to use!")
            print("You can now run the full phenology analysis.")
            
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {str(e)}")
            print("\nPlease authenticate first:")
            print("import ee")
            print("ee.Authenticate()")
            print("ee.Initialize()")
            
    except ImportError:
        print("❌ Earth Engine not installed")
        print("\nPlease install it first:")
        print("pip install earthengine-api geemap")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def main():
    """Main function."""
    
    print("Welcome to the Google Earth Engine Setup Guide!")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Show setup instructions")
        print("2. Test Earth Engine installation")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print_setup_instructions()
        elif choice == '2':
            test_earth_engine()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()

