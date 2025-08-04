import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sales_data(n_records=10000):
    """
    Generate comprehensive sales data with multiple dimensions
    """
    np.random.seed(42)
    random.seed(42)
    
    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories and products
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Automotive', 'Toys']
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera', 'Smartwatch'],
        'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Sweater'],
        'Home & Garden': ['Furniture', 'Kitchen Appliances', 'Garden Tools', 'Lighting', 'Decor'],
        'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Running Shoes', 'Gym Equipment'],
        'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Cookbook', 'Biography', 'Children'],
        'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Perfume', 'Bath Products'],
        'Automotive': ['Car Parts', 'Accessories', 'Tools', 'Maintenance', 'Electronics'],
        'Toys': ['Board Games', 'Action Figures', 'Puzzles', 'Educational Toys', 'Outdoor Toys']
    }
    
    # Regions and cities
    regions = {
        'North America': ['New York', 'Los Angeles', 'Chicago', 'Toronto', 'Vancouver'],
        'Europe': ['London', 'Paris', 'Berlin', 'Madrid', 'Rome'],
        'Asia': ['Tokyo', 'Singapore', 'Seoul', 'Hong Kong', 'Bangkok'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']
    }
    
    # Customer segments
    customer_segments = ['Premium', 'Regular', 'Budget', 'VIP']
    
    # Generate data
    data = []
    
    for _ in range(n_records):
        # Random date
        date = random.choice(date_range)
        
        # Random category and product
        category = random.choice(categories)
        product = random.choice(products[category])
        
        # Random region and city
        region = random.choice(list(regions.keys()))
        city = random.choice(regions[region])
        
        # Customer segment
        customer_segment = random.choice(customer_segments)
        
        # Price based on category and segment
        base_prices = {
            'Electronics': (200, 2000),
            'Clothing': (20, 200),
            'Home & Garden': (50, 500),
            'Sports': (30, 300),
            'Books': (10, 50),
            'Beauty': (15, 150),
            'Automotive': (100, 1000),
            'Toys': (20, 100)
        }
        
        min_price, max_price = base_prices[category]
        price_multipliers = {'Premium': 1.5, 'Regular': 1.0, 'Budget': 0.7, 'VIP': 2.0}
        base_price = random.uniform(min_price, max_price)
        price = base_price * price_multipliers[customer_segment]
        
        # Quantity (higher for budget, lower for premium)
        quantity_multipliers = {'Premium': 1, 'Regular': 2, 'Budget': 3, 'VIP': 1}
        quantity = random.randint(1, 5) * quantity_multipliers[customer_segment]
        
        # Total amount
        total_amount = price * quantity
        
        # Discount based on quantity and segment
        discount_rate = 0
        if quantity >= 3:
            discount_rate = 0.1
        if customer_segment == 'VIP':
            discount_rate += 0.05
        
        discount_amount = total_amount * discount_rate
        final_amount = total_amount - discount_amount
        
        # Customer ID
        customer_id = f"CUST_{random.randint(1000, 9999)}"
        
        # Sales channel
        sales_channels = ['Online', 'In-Store', 'Mobile App', 'Phone']
        sales_channel = random.choice(sales_channels)
        
        # Payment method
        payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet', 'Bank Transfer']
        payment_method = random.choice(payment_methods)
        
        # Add seasonal effects
        month = date.month
        seasonal_multiplier = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.3
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.1
        
        final_amount *= seasonal_multiplier
        
        data.append({
            'Date': date,
            'Customer_ID': customer_id,
            'Customer_Segment': customer_segment,
            'Region': region,
            'City': city,
            'Category': category,
            'Product': product,
            'Price': round(price, 2),
            'Quantity': quantity,
            'Total_Amount': round(total_amount, 2),
            'Discount_Rate': round(discount_rate, 3),
            'Discount_Amount': round(discount_amount, 2),
            'Final_Amount': round(final_amount, 2),
            'Sales_Channel': sales_channel,
            'Payment_Method': payment_method,
            'Year': date.year,
            'Month': date.month,
            'Quarter': (date.month - 1) // 3 + 1
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate data
    print("Generating sales data...")
    sales_data = generate_sales_data(50000)  # 50K records
    
    # Save to CSV
    sales_data.to_csv('sales_data.csv', index=False)
    print(f"Generated {len(sales_data)} sales records")
    print("Data saved to 'sales_data.csv'")
    
    # Display sample
    print("\nSample data:")
    print(sales_data.head())
    
    # Basic statistics
    print(f"\nTotal Revenue: ${sales_data['Final_Amount'].sum():,.2f}")
    print(f"Average Order Value: ${sales_data['Final_Amount'].mean():.2f}")
    print(f"Total Orders: {len(sales_data)}") 