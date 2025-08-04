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

class SalesDashboard:
    def __init__(self, data_path='processed_sales_data.csv'):
        self.data_path = data_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load processed sales data"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def display_header(self):
        """Display dashboard header"""
        st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def display_kpi_metrics(self):
        """Display key performance indicators"""
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = self.data['Final_Amount'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${total_revenue:,.0f}</div>
                <div class="metric-label">Total Revenue</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_orders = len(self.data)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_orders:,}</div>
                <div class="metric-label">Total Orders</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_order_value = self.data['Final_Amount'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${avg_order_value:.2f}</div>
                <div class="metric-label">Average Order Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_customers = self.data['Customer_ID'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_customers:,}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
    
    def display_revenue_trends(self):
        """Display revenue trends over time"""
        st.subheader("📈 Revenue Trends")
        
        # Monthly revenue trend
        monthly_revenue = self.data.groupby(self.data['Date'].dt.to_period('M'))['Final_Amount'].sum().reset_index()
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
    
    def display_category_analysis(self):
        """Display category performance analysis"""
        st.subheader("🏷️ Category Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category revenue
            category_revenue = self.data.groupby('Category')['Final_Amount'].sum().sort_values(ascending=False)
            
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
            category_orders = self.data.groupby('Category')['Final_Amount'].count().sort_values(ascending=False)
            
            fig = px.pie(
                values=category_orders.values,
                names=category_orders.index,
                title='Orders by Category'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def display_regional_analysis(self):
        """Display regional performance analysis"""
        st.subheader("🌍 Regional Performance")
        
        # Regional revenue map
        region_revenue = self.data.groupby('Region')['Final_Amount'].sum().reset_index()
        
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
            region_customers = self.data.groupby('Region')['Customer_ID'].nunique().reset_index()
            
            fig = px.bar(
                region_customers,
                x='Region',
                y='Customer_ID',
                title='Customers by Region',
                labels={'Customer_ID': 'Number of Customers'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top cities
            city_revenue = self.data.groupby('City')['Final_Amount'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=city_revenue.values,
                y=city_revenue.index,
                orientation='h',
                title='Top 10 Cities by Revenue',
                labels={'x': 'Revenue ($)', 'y': 'City'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_customer_analysis(self):
        """Display customer segment analysis"""
        st.subheader("👥 Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segment revenue
            segment_revenue = self.data.groupby('Customer_Segment')['Final_Amount'].sum()
            
            fig = px.pie(
                values=segment_revenue.values,
                names=segment_revenue.index,
                title='Revenue by Customer Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer value tiers
            value_tiers = self.data['Customer_Value_Tier'].value_counts()
            
            fig = px.bar(
                x=value_tiers.index,
                y=value_tiers.values,
                title='Customer Distribution by Value Tier',
                labels={'y': 'Number of Customers', 'x': 'Value Tier'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_sales_channel_analysis(self):
        """Display sales channel performance"""
        st.subheader("🛒 Sales Channel Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Channel revenue
            channel_revenue = self.data.groupby('Sales_Channel')['Final_Amount'].sum()
            
            fig = px.bar(
                x=channel_revenue.index,
                y=channel_revenue.values,
                title='Revenue by Sales Channel',
                labels={'y': 'Revenue ($)', 'x': 'Sales Channel'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment method analysis
            payment_revenue = self.data.groupby('Payment_Method')['Final_Amount'].sum()
            
            fig = px.pie(
                values=payment_revenue.values,
                names=payment_revenue.index,
                title='Revenue by Payment Method'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_product_analysis(self):
        """Display product performance analysis"""
        st.subheader("📦 Product Performance")
        
        # Top products
        top_products = self.data.groupby('Product')['Final_Amount'].sum().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title='Top 15 Products by Revenue',
            labels={'x': 'Revenue ($)', 'y': 'Product'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_seasonal_analysis(self):
        """Display seasonal patterns"""
        st.subheader("📅 Seasonal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly patterns
            monthly_pattern = self.data.groupby('Month')['Final_Amount'].sum()
            
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
                'Spring': self.data[self.data['Month'].isin([3, 4, 5])]['Final_Amount'].sum(),
                'Summer': self.data[self.data['Month'].isin([6, 7, 8])]['Final_Amount'].sum(),
                'Fall': self.data[self.data['Month'].isin([9, 10, 11])]['Final_Amount'].sum(),
                'Winter': self.data[self.data['Month'].isin([12, 1, 2])]['Final_Amount'].sum()
            }
            
            fig = px.bar(
                x=list(seasonal_data.keys()),
                y=list(seasonal_data.values()),
                title='Revenue by Season',
                labels={'y': 'Revenue ($)', 'x': 'Season'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_profit_analysis(self):
        """Display profit margin analysis"""
        st.subheader("💰 Profit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit margin distribution
            fig = px.histogram(
                self.data,
                x='Profit_Margin',
                nbins=30,
                title='Profit Margin Distribution',
                labels={'Profit_Margin': 'Profit Margin (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category profit margins
            category_profit = self.data.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=category_profit.index,
                y=category_profit.values,
                title='Average Profit Margin by Category',
                labels={'y': 'Profit Margin (%)', 'x': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_interactive_filters(self):
        """Display interactive filters"""
        st.sidebar.subheader("🔍 Filters")
        
        # Date range filter
        min_date = self.data['Date'].min()
        max_date = self.data['Date'].max()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        # Category filter
        categories = ['All'] + list(self.data['Category'].unique())
        selected_category = st.sidebar.selectbox("Select Category", categories)
        
        # Region filter
        regions = ['All'] + list(self.data['Region'].unique())
        selected_region = st.sidebar.selectbox("Select Region", regions)
        
        # Customer segment filter
        segments = ['All'] + list(self.data['Customer_Segment'].unique())
        selected_segment = st.sidebar.selectbox("Select Customer Segment", segments)
        
        return {
            'date_range': date_range,
            'category': selected_category,
            'region': selected_region,
            'segment': selected_segment
        }
    
    def apply_filters(self, filters):
        """Apply filters to data"""
        filtered_data = self.data.copy()
        
        # Date filter
        if len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            filtered_data = filtered_data[
                (filtered_data['Date'].dt.date >= start_date) &
                (filtered_data['Date'].dt.date <= end_date)
            ]
        
        # Category filter
        if filters['category'] != 'All':
            filtered_data = filtered_data[filtered_data['Category'] == filters['category']]
        
        # Region filter
        if filters['region'] != 'All':
            filtered_data = filtered_data[filtered_data['Region'] == filters['region']]
        
        # Segment filter
        if filters['segment'] != 'All':
            filtered_data = filtered_data[filtered_data['Customer_Segment'] == filters['segment']]
        
        return filtered_data
    
    def display_summary_statistics(self):
        """Display summary statistics"""
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Revenue Statistics**")
            revenue_stats = self.data['Final_Amount'].describe()
            st.write(revenue_stats)
        
        with col2:
            st.write("**Order Statistics**")
            order_stats = self.data['Quantity'].describe()
            st.write(order_stats)
        
        with col3:
            st.write("**Profit Margin Statistics**")
            profit_stats = self.data['Profit_Margin'].describe()
            st.write(profit_stats)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        if self.data is None:
            st.error("Failed to load data. Please check the data file.")
            return
        
        # Display header
        self.display_header()
        
        # Display filters
        filters = self.display_interactive_filters()
        
        # Apply filters
        filtered_data = self.apply_filters(filters)
        
        # Update data with filtered data
        original_data = self.data
        self.data = filtered_data
        
        # Display KPI metrics
        self.display_kpi_metrics()
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Revenue Trends", "🏷️ Categories", "🌍 Regions", 
            "👥 Customers", "🛒 Channels", "📦 Products", "📊 Statistics"
        ])
        
        with tab1:
            self.display_revenue_trends()
            self.display_seasonal_analysis()
        
        with tab2:
            self.display_category_analysis()
        
        with tab3:
            self.display_regional_analysis()
        
        with tab4:
            self.display_customer_analysis()
        
        with tab5:
            self.display_sales_channel_analysis()
        
        with tab6:
            self.display_product_analysis()
        
        with tab7:
            self.display_profit_analysis()
            self.display_summary_statistics()
        
        # Restore original data
        self.data = original_data

def main():
    """Main function to run the dashboard"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🚀 Advanced Sales Analytics Dashboard</h2>
        <p>Comprehensive analysis of sales data with interactive visualizations and neural network insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = SalesDashboard()
    
    # Run dashboard
    dashboard.run_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>📊 Sales Analytics Dashboard | Built with Streamlit & Plotly | Neural Network Powered</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 