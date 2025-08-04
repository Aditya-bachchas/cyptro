#!/usr/bin/env python3
"""
Simple Data Generation
Generates sales data for the dashboard.
"""

from data_generator import generate_sales_data
from data_processor import SalesDataProcessor

def main():
    """Generate and process data"""
    print("📊 Generating Sales Data...")
    
    # Generate data
    sales_data = generate_sales_data(10000)  # 10K records
    sales_data.to_csv('sales_data.csv', index=False)
    
    print(f"✅ Generated {len(sales_data)} sales records")
    print(f"✅ Data saved to 'sales_data.csv'")
    
    # Process data
    print("\n🔄 Processing Data...")
    processor = SalesDataProcessor()
    processor.process_pipeline()
    
    print("✅ Data processing completed!")
    print("✅ Processed data saved to 'processed_sales_data.csv'")
    
    # Show summary
    print(f"\n📈 Summary:")
    print(f"  - Total Revenue: ${sales_data['Final_Amount'].sum():,.2f}")
    print(f"  - Average Order Value: ${sales_data['Final_Amount'].mean():.2f}")
    print(f"  - Total Orders: {len(sales_data)}")
    print(f"  - Unique Customers: {sales_data['Customer_ID'].nunique()}")
    
    print("\n🎉 Data generation completed! You can now run the dashboard.")

if __name__ == "__main__":
    main() 