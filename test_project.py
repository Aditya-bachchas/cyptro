#!/usr/bin/env python3
"""
Test script for Sales Analytics Data Science Project
Verifies all components work correctly before running the full pipeline.
"""

import sys
import os
import time
from datetime import datetime

def test_data_generation():
    """Test data generation functionality"""
    print("🧪 Testing Data Generation...")
    
    try:
        from data_generator import generate_sales_data
        
        # Generate small test dataset
        test_data = generate_sales_data(1000)  # 1K records for testing
        
        print(f"✅ Generated {len(test_data)} test records")
        print(f"✅ Columns: {list(test_data.columns)}")
        print(f"✅ Revenue: ${test_data['Final_Amount'].sum():,.2f}")
        print(f"✅ Customers: {test_data['Customer_ID'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("\n🧪 Testing Data Processing...")
    
    try:
        from data_processor import SalesDataProcessor
        
        # Create test data first
        from data_generator import generate_sales_data
        test_data = generate_sales_data(1000)
        test_data.to_csv('test_sales_data.csv', index=False)
        
        # Test processor
        processor = SalesDataProcessor('test_sales_data.csv')
        
        # Test each step
        if not processor.load_data():
            return False
        
        if not processor.clean_data():
            return False
        
        if not processor.create_features():
            return False
        
        if not processor.calculate_metrics():
            return False
        
        print("✅ Data processing pipeline works correctly")
        print(f"✅ Processed {len(processor.processed_data)} records")
        print(f"✅ Generated {len(processor.metrics)} metrics")
        
        # Clean up test file
        if os.path.exists('test_sales_data.csv'):
            os.remove('test_sales_data.csv')
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_neural_network():
    """Test machine learning functionality"""
    print("\n🧪 Testing Machine Learning...")
    
    try:
        from neural_network_simple import SalesNeuralNetwork
        
        # Create test processed data first
        from data_generator import generate_sales_data
        from data_processor import SalesDataProcessor
        
        test_data = generate_sales_data(1000)
        test_data.to_csv('test_sales_data.csv', index=False)
        
        processor = SalesDataProcessor('test_sales_data.csv')
        processor.process_pipeline()
        processor.save_processed_data('test_processed_data.csv')
        
        # Test neural network
        nn_model = SalesNeuralNetwork('test_processed_data.csv')
        
        if not nn_model.load_data():
            return False
        
        if not nn_model.prepare_features():
            return False
        
        if not nn_model.build_model(architecture='simple'):
            return False
        
        print("✅ Machine learning model built successfully")
        print(f"✅ Features: {len(nn_model.feature_columns)}")
        print(f"✅ Training samples: {len(nn_model.X_train)}")
        print(f"✅ Test samples: {len(nn_model.X_test)}")
        
        # Clean up test files
        for file in ['test_sales_data.csv', 'test_processed_data.csv']:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ Machine learning test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print("\n🧪 Testing Dashboard...")
    
    try:
        # Create test data for dashboard
        from data_generator import generate_sales_data
        from data_processor import SalesDataProcessor
        
        test_data = generate_sales_data(1000)
        test_data.to_csv('test_sales_data.csv', index=False)
        
        processor = SalesDataProcessor('test_sales_data.csv')
        processor.process_pipeline()
        processor.save_processed_data('test_processed_data.csv')
        
        # Test dashboard initialization
        from dashboard import SalesDashboard
        
        dashboard = SalesDashboard('test_processed_data.csv')
        
        if dashboard.data is not None:
            print("✅ Dashboard data loading works")
            print(f"✅ Loaded {len(dashboard.data)} records")
            print("✅ Dashboard components ready")
        else:
            print("❌ Dashboard data loading failed")
            return False
        
        # Clean up test files
        for file in ['test_sales_data.csv', 'test_processed_data.csv']:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'streamlit', 'scikit-learn', 'tensorflow', 'keras'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def main():
    """Run all tests"""
    print("🧪 SALES ANALYTICS PROJECT - COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Generation", test_data_generation),
        ("Data Processing", test_data_processing),
        ("Machine Learning", test_neural_network),
        ("Dashboard", test_dashboard)
    ]
    
    results = []
    
    for test_name, test_function in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        success = test_function()
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append((test_name, success, duration))
        
        if success:
            print(f"✅ {test_name} test passed! ({duration:.2f}s)")
        else:
            print(f"❌ {test_name} test failed! ({duration:.2f}s)")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<20} ({duration:.2f}s)")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to run.")
        print("🚀 Run 'python main.py' to start the complete pipeline.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 