import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesDataProcessor:
    def __init__(self, data_path='sales_data.csv'):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.metrics = {}
        
    def load_data(self):
        """Load raw sales data"""
        try:
            self.raw_data = pd.read_csv(self.data_path)
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
            print(f"Loaded {len(self.raw_data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean and validate the data"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        # Remove duplicates
        initial_count = len(self.raw_data)
        self.raw_data = self.raw_data.drop_duplicates()
        
        # Remove rows with negative amounts
        self.raw_data = self.raw_data[self.raw_data['Final_Amount'] > 0]
        
        # Remove rows with missing critical data
        self.raw_data = self.raw_data.dropna(subset=['Customer_ID', 'Category', 'Final_Amount'])
        
        # Validate date range
        self.raw_data = self.raw_data[
            (self.raw_data['Date'] >= '2022-01-01') & 
            (self.raw_data['Date'] <= '2024-12-31')
        ]
        
        # Validate quantities
        self.raw_data = self.raw_data[self.raw_data['Quantity'] > 0]
        
        print(f"Data cleaning completed:")
        print(f"  - Removed {initial_count - len(self.raw_data)} invalid records")
        print(f"  - Final dataset: {len(self.raw_data)} records")
        
        return True
    
    def create_features(self):
        """Create additional features for analysis"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        df = self.raw_data.copy()
        
        # Time-based features
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        df['Is_Weekend'] = df['Date'].dt.weekday >= 5
        df['Is_Holiday_Season'] = df['Month'].isin([11, 12])
        df['Is_Summer'] = df['Month'].isin([6, 7, 8])
        
        # Customer features
        customer_stats = df.groupby('Customer_ID').agg({
            'Final_Amount': ['sum', 'count', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        customer_stats.columns = ['Total_Spent', 'Order_Count', 'Avg_Order_Value', 'First_Order', 'Last_Order']
        customer_stats['Customer_Lifetime_Days'] = (customer_stats['Last_Order'] - customer_stats['First_Order']).dt.days
        
        # Merge customer features back
        df = df.merge(customer_stats, left_on='Customer_ID', right_index=True, how='left')
        
        # Product features
        product_stats = df.groupby('Product').agg({
            'Final_Amount': ['sum', 'count', 'mean'],
            'Quantity': 'sum'
        }).round(2)
        product_stats.columns = ['Product_Revenue', 'Product_Orders', 'Product_Avg_Price', 'Product_Quantity_Sold']
        
        # Merge product features back
        df = df.merge(product_stats, left_on='Product', right_index=True, how='left')
        
        # Category features
        category_stats = df.groupby('Category').agg({
            'Final_Amount': ['sum', 'count', 'mean'],
            'Quantity': 'sum'
        }).round(2)
        category_stats.columns = ['Category_Revenue', 'Category_Orders', 'Category_Avg_Price', 'Category_Quantity_Sold']
        
        # Merge category features back
        df = df.merge(category_stats, left_on='Category', right_index=True, how='left')
        
        # Regional features
        region_stats = df.groupby('Region').agg({
            'Final_Amount': ['sum', 'count', 'mean'],
            'Customer_ID': 'nunique'
        }).round(2)
        region_stats.columns = ['Region_Revenue', 'Region_Orders', 'Region_Avg_Order', 'Region_Customers']
        
        # Merge regional features back
        df = df.merge(region_stats, left_on='Region', right_index=True, how='left')
        
        # Sales channel features
        channel_stats = df.groupby('Sales_Channel').agg({
            'Final_Amount': ['sum', 'count', 'mean']
        }).round(2)
        channel_stats.columns = ['Channel_Revenue', 'Channel_Orders', 'Channel_Avg_Order']
        
        # Merge channel features back
        df = df.merge(channel_stats, left_on='Sales_Channel', right_index=True, how='left')
        
        # Payment method features
        payment_stats = df.groupby('Payment_Method').agg({
            'Final_Amount': ['sum', 'count', 'mean']
        }).round(2)
        payment_stats.columns = ['Payment_Revenue', 'Payment_Orders', 'Payment_Avg_Order']
        
        # Merge payment features back
        df = df.merge(payment_stats, left_on='Payment_Method', right_index=True, how='left')
        
        # Customer segment features
        segment_stats = df.groupby('Customer_Segment').agg({
            'Final_Amount': ['sum', 'count', 'mean'],
            'Customer_ID': 'nunique'
        }).round(2)
        segment_stats.columns = ['Segment_Revenue', 'Segment_Orders', 'Segment_Avg_Order', 'Segment_Customers']
        
        # Merge segment features back
        df = df.merge(segment_stats, left_on='Customer_Segment', right_index=True, how='left')
        
        # Profit margin estimation (simplified)
        df['Estimated_Cost'] = df['Price'] * 0.6  # Assume 40% margin
        df['Estimated_Profit'] = df['Final_Amount'] - (df['Estimated_Cost'] * df['Quantity'])
        df['Profit_Margin'] = (df['Estimated_Profit'] / df['Final_Amount']) * 100
        
        # Customer value tier
        df['Customer_Value_Tier'] = pd.cut(
            df['Total_Spent'], 
            bins=[0, 100, 500, 1000, float('inf')], 
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )
        
        self.processed_data = df
        print("Feature engineering completed")
        return True
    
    def calculate_metrics(self):
        """Calculate comprehensive business metrics"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return False
        
        df = self.processed_data
        
        # Overall metrics
        self.metrics['total_revenue'] = df['Final_Amount'].sum()
        self.metrics['total_orders'] = len(df)
        self.metrics['total_customers'] = df['Customer_ID'].nunique()
        self.metrics['average_order_value'] = df['Final_Amount'].mean()
        self.metrics['average_profit_margin'] = df['Profit_Margin'].mean()
        self.metrics['total_profit'] = df['Estimated_Profit'].sum()
        
        # Time-based metrics
        self.metrics['revenue_by_year'] = df.groupby('Year')['Final_Amount'].sum().to_dict()
        self.metrics['revenue_by_month'] = df.groupby('Month')['Final_Amount'].sum().to_dict()
        self.metrics['revenue_by_quarter'] = df.groupby('Quarter')['Final_Amount'].sum().to_dict()
        
        # Category metrics
        self.metrics['category_revenue'] = df.groupby('Category')['Final_Amount'].sum().sort_values(ascending=False).to_dict()
        self.metrics['category_orders'] = df.groupby('Category')['Final_Amount'].count().sort_values(ascending=False).to_dict()
        
        # Regional metrics
        self.metrics['region_revenue'] = df.groupby('Region')['Final_Amount'].sum().sort_values(ascending=False).to_dict()
        self.metrics['region_customers'] = df.groupby('Region')['Customer_ID'].nunique().sort_values(ascending=False).to_dict()
        
        # Customer segment metrics
        self.metrics['segment_revenue'] = df.groupby('Customer_Segment')['Final_Amount'].sum().to_dict()
        self.metrics['segment_customers'] = df.groupby('Customer_Segment')['Customer_ID'].nunique().to_dict()
        
        # Sales channel metrics
        self.metrics['channel_revenue'] = df.groupby('Sales_Channel')['Final_Amount'].sum().to_dict()
        self.metrics['channel_orders'] = df.groupby('Sales_Channel')['Final_Amount'].count().to_dict()
        
        # Product performance
        self.metrics['top_products'] = df.groupby('Product')['Final_Amount'].sum().sort_values(ascending=False).head(10).to_dict()
        self.metrics['top_categories'] = df.groupby('Category')['Final_Amount'].sum().sort_values(ascending=False).to_dict()
        
        # Customer value metrics
        self.metrics['customer_value_tiers'] = df['Customer_Value_Tier'].value_counts().to_dict()
        
        # Growth metrics (monthly)
        monthly_revenue = df.groupby([df['Date'].dt.to_period('M')])['Final_Amount'].sum()
        self.metrics['monthly_revenue'] = monthly_revenue.to_dict()
        
        # Seasonal analysis
        self.metrics['seasonal_revenue'] = {
            'Spring': df[df['Month'].isin([3, 4, 5])]['Final_Amount'].sum(),
            'Summer': df[df['Month'].isin([6, 7, 8])]['Final_Amount'].sum(),
            'Fall': df[df['Month'].isin([9, 10, 11])]['Final_Amount'].sum(),
            'Winter': df[df['Month'].isin([12, 1, 2])]['Final_Amount'].sum()
        }
        
        print("Metrics calculation completed")
        return True
    
    def get_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.metrics:
            print("No metrics available. Please calculate metrics first.")
            return None
        
        report = f"""
        ========================================
        SALES ANALYTICS SUMMARY REPORT
        ========================================
        
        OVERALL PERFORMANCE:
        - Total Revenue: ${self.metrics['total_revenue']:,.2f}
        - Total Orders: {self.metrics['total_orders']:,}
        - Total Customers: {self.metrics['total_customers']:,}
        - Average Order Value: ${self.metrics['average_order_value']:.2f}
        - Average Profit Margin: {self.metrics['average_profit_margin']:.1f}%
        - Total Estimated Profit: ${self.metrics['total_profit']:,.2f}
        
        TOP PERFORMING CATEGORIES:
        """
        
        for i, (category, revenue) in enumerate(list(self.metrics['category_revenue'].items())[:5], 1):
            report += f"{i}. {category}: ${revenue:,.2f}\n"
        
        report += f"""
        TOP PERFORMING REGIONS:
        """
        
        for i, (region, revenue) in enumerate(list(self.metrics['region_revenue'].items())[:5], 1):
            report += f"{i}. {region}: ${revenue:,.2f}\n"
        
        report += f"""
        CUSTOMER SEGMENT ANALYSIS:
        """
        
        for segment, revenue in self.metrics['segment_revenue'].items():
            customers = self.metrics['segment_customers'][segment]
            avg_revenue = revenue / customers
            report += f"- {segment}: ${revenue:,.2f} ({customers} customers, avg: ${avg_revenue:.2f})\n"
        
        return report
    
    def save_processed_data(self, filename='processed_sales_data.csv'):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
            print(f"Processed data saved to {filename}")
        else:
            print("No processed data available")
    
    def process_pipeline(self):
        """Run the complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        if not self.create_features():
            return False
        
        if not self.calculate_metrics():
            return False
        
        self.save_processed_data()
        
        print("Data processing pipeline completed successfully!")
        return True

if __name__ == "__main__":
    # Run the complete pipeline
    processor = SalesDataProcessor()
    processor.process_pipeline()
    
    # Print summary report
    report = processor.get_summary_report()
    if report:
        print(report) 