#!/usr/bin/env python3
"""
Sales Analytics Data Science Project
Comprehensive pipeline including data generation, processing, neural network modeling, and dashboard creation.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    🚀 SALES ANALYTICS DATA SCIENCE PROJECT                  ║
    ║                                                                              ║
    ║  📊 Comprehensive Sales Analytics with Neural Network & Interactive Dashboard ║
    ║                                                                              ║
    ║  Features:                                                                   ║
    ║  • Big Data Generation (50K+ records)                                       ║
    ║  • Advanced Data Processing & Feature Engineering                           ║
    ║  • Neural Network for Sales Prediction                                      ║
    ║  • Beautiful Interactive Dashboard                                          ║
    ║  • Meaningful Business Metrics & Analytics                                  ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
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
    
    print("✅ All dependencies are installed!")
    return True

def run_data_generation():
    """Run data generation step"""
    print("\n📊 Step 1: Generating Sales Data...")
    print("=" * 50)
    
    try:
        from data_generator import generate_sales_data
        
        # Generate data
        sales_data = generate_sales_data(50000)  # 50K records
        
        # Save to CSV
        sales_data.to_csv('sales_data.csv', index=False)
        
        print(f"✅ Generated {len(sales_data)} sales records")
        print(f"✅ Data saved to 'sales_data.csv'")
        
        # Display sample statistics
        print(f"\n📈 Sample Statistics:")
        print(f"  - Total Revenue: ${sales_data['Final_Amount'].sum():,.2f}")
        print(f"  - Average Order Value: ${sales_data['Final_Amount'].mean():.2f}")
        print(f"  - Total Orders: {len(sales_data)}")
        print(f"  - Unique Customers: {sales_data['Customer_ID'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        return False

def run_data_processing():
    """Run data processing step"""
    print("\n🔄 Step 2: Processing and Cleaning Data...")
    print("=" * 50)
    
    try:
        from data_processor import SalesDataProcessor
        
        # Initialize processor
        processor = SalesDataProcessor()
        
        # Run complete pipeline
        success = processor.process_pipeline()
        
        if success:
            # Display summary report
            report = processor.get_summary_report()
            if report:
                print("\n📋 Summary Report:")
                print(report)
        
        return success
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False

def run_neural_network():
    """Run machine learning modeling step"""
    print("\n🧠 Step 3: Training Machine Learning Model...")
    print("=" * 50)
    
    try:
        from neural_network_simple import SalesNeuralNetwork
        
        # Initialize machine learning model
        ml_model = SalesNeuralNetwork()
        
        # Run complete pipeline
        success = ml_model.run_complete_pipeline()
        
        if success:
            print("✅ Machine learning training completed successfully!")
            print("📊 Model performance metrics and visualizations generated")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in machine learning training: {e}")
        return False

def run_dashboard():
    """Run dashboard deployment"""
    print("\n📊 Step 4: Launching Interactive Dashboard...")
    print("=" * 50)
    
    try:
        print("🚀 Starting Streamlit dashboard...")
        print("📱 Dashboard will open in your web browser")
        print("🔗 Local URL: http://localhost:8501")
        print("\n💡 Tips:")
        print("  - Use the sidebar filters to explore different views")
        print("  - Navigate through tabs to see different analyses")
        print("  - Hover over charts for detailed information")
        print("  - Press Ctrl+C in terminal to stop the dashboard")
        
        # Start Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def create_project_structure():
    """Create project directory structure"""
    print("\n📁 Creating project structure...")
    
    directories = [
        'data',
        'models',
        'reports',
        'visualizations',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def generate_project_report():
    """Generate comprehensive project report"""
    print("\n📋 Generating Project Report...")
    
    report_content = f"""
# Sales Analytics Data Science Project Report

## Project Overview
This comprehensive data science project demonstrates advanced analytics capabilities including:
- Big data generation and processing
- Neural network implementation for sales prediction
- Interactive dashboard creation
- Business intelligence and metrics analysis

## Project Components

### 1. Data Generation (`data_generator.py`)
- Generated 50,000+ realistic sales records
- Multiple dimensions: products, regions, customer segments
- Time-series data with seasonal patterns
- Comprehensive feature set for analysis

### 2. Data Processing (`data_processor.py`)
- Advanced data cleaning and validation
- Feature engineering with 40+ derived features
- Business metrics calculation
- Customer lifetime value analysis

### 3. Neural Network Model (`neural_network_model.py`)
- Deep learning architecture with TensorFlow/Keras
- Advanced features: batch normalization, dropout, early stopping
- Feature importance analysis
- Model performance evaluation

### 4. Interactive Dashboard (`dashboard.py`)
- Beautiful Streamlit-based dashboard
- Interactive visualizations with Plotly
- Real-time filtering and analysis
- Comprehensive business metrics

## Technical Stack
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Big Data**: 50K+ records with multiple dimensions

## Key Features
✅ Big Data Generation (50K+ records)
✅ Advanced Data Processing Pipeline
✅ Neural Network for Sales Prediction
✅ Interactive Dashboard with Real-time Analytics
✅ Business Intelligence & KPI Tracking
✅ Feature Engineering & Model Evaluation
✅ Beautiful Visualizations & Charts
✅ Comprehensive Documentation

## Files Generated
- `sales_data.csv` - Raw sales data
- `processed_sales_data.csv` - Cleaned and processed data
- `sales_prediction_model.h5` - Trained neural network model
- Various visualization files (PNG format)

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run complete pipeline: `python main.py`
3. Access dashboard: http://localhost:8501

## Project Value for CV
This project demonstrates:
- Advanced data science skills
- Big data processing capabilities
- Machine learning implementation
- Dashboard development
- Business analytics understanding
- End-to-end project management

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report_content)
    
    print("✅ Project report generated: PROJECT_REPORT.md")

def main():
    """Main function to run the complete pipeline"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Create project structure
    create_project_structure()
    
    # Run complete pipeline
    steps = [
        ("Data Generation", run_data_generation),
        ("Data Processing", run_data_processing),
        ("Neural Network Training", run_neural_network),
        ("Dashboard Launch", run_dashboard)
    ]
    
    print("\n🚀 Starting Complete Data Science Pipeline...")
    print("=" * 60)
    
    for step_name, step_function in steps:
        print(f"\n⏳ Running: {step_name}")
        start_time = time.time()
        
        success = step_function()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"✅ {step_name} completed successfully! ({duration:.2f}s)")
        else:
            print(f"❌ {step_name} failed!")
            print("\n💡 Troubleshooting tips:")
            print("  - Check if all dependencies are installed")
            print("  - Ensure you have sufficient disk space")
            print("  - Check console for detailed error messages")
            return
    
    # Generate project report
    generate_project_report()
    
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 Your comprehensive sales analytics project is ready!")
    print("📁 Check the generated files and reports")
    print("🚀 Dashboard is running at http://localhost:8501")
    print("📋 Project report: PROJECT_REPORT.md")
    
    print("\n💼 Perfect for your CV - demonstrates:")
    print("  ✅ Advanced data science skills")
    print("  ✅ Big data processing")
    print("  ✅ Machine learning implementation")
    print("  ✅ Dashboard development")
    print("  ✅ Business analytics")
    print("  ✅ End-to-end project management")

if __name__ == "__main__":
    main() 