#!/usr/bin/env python3
"""
Sales Analytics Dashboard - Streamlit App
Main entry point for Streamlit Community Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    n_records = 5000
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Automotive', 'Toys']
    regions = ['North America', 'Europe', 'Asia', 'Australia']
    customer_segments = ['Premium', 'Regular', 'Budget', 'VIP']
    sales_channels = ['Online', 'In-Store', 'Mobile App', 'Phone']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet', 'Bank Transfer']
    
    data = []
    for i in range(n_records):
        date = np.random.choice(dates)
        category = np.random.choice(categories)
        region = np.random.choice(regions)
        customer_segment = np.random.choice(customer_segments)
        sales_channel = np.random.choice(sales_channels)
        payment_method = np.random.choice(payment_methods)
        
        # Price based on category
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
        price = np.random.uniform(min_price, max_price)
        quantity = np.random.randint(1, 5)
        total_amount = price * quantity
        
        # Discount
        discount_rate = 0
        if quantity >= 3:
            discount_rate = 0.1
        if customer_segment == 'VIP':
            discount_rate += 0.05
        
        discount_amount = total_amount * discount_rate
        final_amount = total_amount - discount_amount
        
        data.append({
            'Date': date,
            'Customer_ID': f"CUST_{np.random.randint(1000, 9999)}",
            'Customer_Segment': customer_segment,
            'Region': region,
            'Category': category,
            'Product': f"{category} Product {np.random.randint(1, 10)}",
            'Price': round(price, 2),
            'Quantity': quantity,
            'Total_Amount': round(total_amount, 2),
            'Discount_Rate': round(discount_rate, 3),
            'Discount_Amount': round(discount_amount, 2),
            'Final_Amount': round(final_amount, 2),
            'Sales_Channel': sales_channel,
            'Payment_Method': payment_method,
            'Year': pd.to_datetime(date).year,
            'Month': pd.to_datetime(date).month,
            'Quarter': (pd.to_datetime(date).month - 1) // 3 + 1
        })
    
    return pd.DataFrame(data)

def display_header():
    """Display dashboard header"""
    st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

def display_kpi_metrics(data):
    """Display key performance indicators"""
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = data['Final_Amount'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-label">Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_orders = len(data)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_order_value = data['Final_Amount'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_order_value:.2f}</div>
            <div class="metric-label">Average Order Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_customers = data['Customer_ID'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)

def display_revenue_trends(data):
    """Display revenue trends over time"""
    st.subheader("📈 Revenue Trends")
    
    # Monthly revenue trend
    monthly_revenue = data.groupby(data['Date'].dt.to_period('M'))['Final_Amount'].sum().reset_index()
    monthly_revenue['Date'] = monthly_revenue['Date'].astype(str)
    
    fig = px.line(
        monthly_revenue, 
        x='Date', 
        y='Final_Amount',
        title='Monthly Revenue Trend',
        labels={'Final_Amount': 'Revenue ($)', 'Date': 'Month'},
        line_shape='linear',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_category_analysis(data):
    """Display category performance analysis"""
    st.subheader("🏷️ Category Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category revenue
        category_revenue = data.groupby('Category')['Final_Amount'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=category_revenue.values,
            y=category_revenue.index,
            orientation='h',
            title='Revenue by Category',
            labels={'x': 'Revenue ($)', 'y': 'Category'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category orders
        category_orders = data.groupby('Category')['Final_Amount'].count().sort_values(ascending=False)
        
        fig = px.pie(
            values=category_orders.values,
            names=category_orders.index,
            title='Orders by Category'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_regional_analysis(data):
    """Display regional performance analysis"""
    st.subheader("🌍 Regional Performance")
    
    # Regional revenue
    region_revenue = data.groupby('Region')['Final_Amount'].sum().reset_index()
    
    fig = px.bar(
        region_revenue,
        x='Region',
        y='Final_Amount',
        title='Revenue by Region',
        labels={'Final_Amount': 'Revenue ($)'},
        color='Final_Amount',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional customer distribution
    col1, col2 = st.columns(2)
    
    with col1:
        region_customers = data.groupby('Region')['Customer_ID'].nunique().reset_index()
        
        fig = px.bar(
            region_customers,
            x='Region',
            y='Customer_ID',
            title='Customers by Region',
            labels={'Customer_ID': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top products
        top_products = data.groupby('Product')['Final_Amount'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title='Top 10 Products by Revenue',
            labels={'x': 'Revenue ($)', 'y': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_customer_analysis(data):
    """Display customer segment analysis"""
    st.subheader("👥 Customer Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segment revenue
        segment_revenue = data.groupby('Customer_Segment')['Final_Amount'].sum()
        
        fig = px.pie(
            values=segment_revenue.values,
            names=segment_revenue.index,
            title='Revenue by Customer Segment'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer segment distribution
        segment_dist = data['Customer_Segment'].value_counts()
        
        fig = px.bar(
            x=segment_dist.index,
            y=segment_dist.values,
            title='Customer Distribution by Segment',
            labels={'y': 'Number of Customers', 'x': 'Customer Segment'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_sales_channel_analysis(data):
    """Display sales channel performance"""
    st.subheader("🛒 Sales Channel Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel revenue
        channel_revenue = data.groupby('Sales_Channel')['Final_Amount'].sum()
        
        fig = px.bar(
            x=channel_revenue.index,
            y=channel_revenue.values,
            title='Revenue by Sales Channel',
            labels={'y': 'Revenue ($)', 'x': 'Sales Channel'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment method analysis
        payment_revenue = data.groupby('Payment_Method')['Final_Amount'].sum()
        
        fig = px.pie(
            values=payment_revenue.values,
            names=payment_revenue.index,
            title='Revenue by Payment Method'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_seasonal_analysis(data):
    """Display seasonal patterns"""
    st.subheader("📅 Seasonal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly patterns
        monthly_pattern = data.groupby('Month')['Final_Amount'].sum()
        
        fig = px.line(
            x=monthly_pattern.index,
            y=monthly_pattern.values,
            title='Monthly Revenue Pattern',
            labels={'x': 'Month', 'y': 'Revenue ($)'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal revenue
        seasonal_data = {
            'Spring': data[data['Month'].isin([3, 4, 5])]['Final_Amount'].sum(),
            'Summer': data[data['Month'].isin([6, 7, 8])]['Final_Amount'].sum(),
            'Fall': data[data['Month'].isin([9, 10, 11])]['Final_Amount'].sum(),
            'Winter': data[data['Month'].isin([12, 1, 2])]['Final_Amount'].sum()
        }
        
        fig = px.bar(
            x=list(seasonal_data.keys()),
            y=list(seasonal_data.values()),
            title='Revenue by Season',
            labels={'y': 'Revenue ($)', 'x': 'Season'}
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the dashboard"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🚀 Advanced Sales Analytics Dashboard</h2>
        <p>Comprehensive analysis of sales data with interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    with st.spinner("Loading data..."):
        data = generate_sample_data()
    
    # Display header
    display_header()
    
    # Display KPI metrics
    display_kpi_metrics(data)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Revenue Trends", "🏷️ Categories", "🌍 Regions", 
        "👥 Customers", "🛒 Channels", "📅 Seasonal"
    ])
    
    with tab1:
        display_revenue_trends(data)
    
    with tab2:
        display_category_analysis(data)
    
    with tab3:
        display_regional_analysis(data)
    
    with tab4:
        display_customer_analysis(data)
    
    with tab5:
        display_sales_channel_analysis(data)
    
    with tab6:
        display_seasonal_analysis(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>📊 Sales Analytics Dashboard | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 