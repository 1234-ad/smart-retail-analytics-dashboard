"""
Smart Retail Analytics Dashboard
===============================

Interactive Streamlit dashboard for retail business intelligence and analytics.
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Smart Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load and cache sample retail data."""
    np.random.seed(42)
    
    # Generate sample data
    n_records = 5000
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2024, 8, 31)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')[:n_records]
    
    data = {
        'transaction_id': range(1, n_records + 1),
        'date': dates,
        'customer_id': np.random.randint(1, 1000, n_records),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'], n_records),
        'product_name': [f'Product_{i}' for i in np.random.randint(1, 200, n_records)],
        'quantity': np.random.randint(1, 5, n_records),
        'unit_price': np.random.uniform(10, 500, n_records),
        'discount_percent': np.random.uniform(0, 25, n_records),
        'sales_channel': np.random.choice(['Online', 'In-Store', 'Mobile App'], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_records)
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent']/100)
    df['profit_margin'] = np.random.uniform(0.1, 0.4, n_records)
    df['profit'] = df['total_amount'] * df['profit_margin']
    
    return df

def create_kpi_metrics(df):
    """Create KPI metrics for the dashboard."""
    total_revenue = df['total_amount'].sum()
    total_profit = df['profit'].sum()
    avg_order_value = df['total_amount'].mean()
    total_customers = df['customer_id'].nunique()
    total_transactions = len(df)
    profit_margin = (total_profit / total_revenue) * 100
    
    return {
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'avg_order_value': avg_order_value,
        'total_customers': total_customers,
        'total_transactions': total_transactions,
        'profit_margin': profit_margin
    }

def create_revenue_trend_chart(df):
    """Create revenue trend chart."""
    daily_revenue = df.groupby(df['date'].dt.date)['total_amount'].sum().reset_index()
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    
    fig = px.line(daily_revenue, x='date', y='total_amount',
                  title='Daily Revenue Trend',
                  labels={'total_amount': 'Revenue ($)', 'date': 'Date'})
    fig.update_layout(height=400)
    return fig

def create_category_performance_chart(df):
    """Create category performance chart."""
    category_stats = df.groupby('product_category').agg({
        'total_amount': 'sum',
        'profit': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue by Category', 'Profit by Category'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=category_stats['product_category'], y=category_stats['total_amount'],
               name='Revenue', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=category_stats['product_category'], y=category_stats['profit'],
               name='Profit', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_customer_segmentation_chart(df):
    """Create customer segmentation analysis."""
    segment_stats = df.groupby('customer_segment').agg({
        'total_amount': ['sum', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    segment_stats.columns = ['Total Revenue', 'Avg Order Value', 'Customer Count']
    segment_stats = segment_stats.reset_index()
    
    fig = px.scatter(segment_stats, x='Customer Count', y='Avg Order Value',
                     size='Total Revenue', color='customer_segment',
                     title='Customer Segmentation Analysis',
                     labels={'customer_segment': 'Segment'})
    fig.update_layout(height=400)
    return fig

def create_regional_performance_chart(df):
    """Create regional performance chart."""
    regional_stats = df.groupby('region')['total_amount'].sum().reset_index()
    
    fig = px.pie(regional_stats, values='total_amount', names='region',
                 title='Revenue Distribution by Region')
    fig.update_layout(height=400)
    return fig

def create_sales_channel_chart(df):
    """Create sales channel performance chart."""
    channel_stats = df.groupby('sales_channel').agg({
        'total_amount': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue by Channel', 'Transactions by Channel'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=channel_stats['sales_channel'], y=channel_stats['total_amount'],
               name='Revenue', marker_color='orange'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=channel_stats['sales_channel'], y=channel_stats['transaction_id'],
               name='Transactions', marker_color='purple'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_hourly_pattern_chart(df):
    """Create hourly sales pattern chart."""
    hourly_sales = df.groupby(df['date'].dt.hour)['total_amount'].sum().reset_index()
    hourly_sales['date'] = hourly_sales['date'].astype(str) + ':00'
    
    fig = px.bar(hourly_sales, x='date', y='total_amount',
                 title='Sales Pattern by Hour of Day',
                 labels={'total_amount': 'Revenue ($)', 'date': 'Hour'})
    fig.update_layout(height=400)
    return fig

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Smart Retail Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_sample_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Select Product Categories",
        options=df['product_category'].unique(),
        default=df['product_category'].unique()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Sales channel filter
    channels = st.sidebar.multiselect(
        "Select Sales Channels",
        options=df['sales_channel'].unique(),
        default=df['sales_channel'].unique()
    )
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['date'].dt.date >= start_date) & 
            (df['date'].dt.date <= end_date) &
            (df['product_category'].isin(categories)) &
            (df['region'].isin(regions)) &
            (df['sales_channel'].isin(channels))
        ]
    else:
        df_filtered = df[
            (df['product_category'].isin(categories)) &
            (df['region'].isin(regions)) &
            (df['sales_channel'].isin(channels))
        ]
    
    # KPI Metrics
    st.header("📈 Key Performance Indicators")
    
    if not df_filtered.empty:
        kpis = create_kpi_metrics(df_filtered)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="💰 Total Revenue",
                value=f"${kpis['total_revenue']:,.0f}",
                delta=f"{kpis['total_revenue']/1000:.1f}K"
            )
        
        with col2:
            st.metric(
                label="💵 Total Profit",
                value=f"${kpis['total_profit']:,.0f}",
                delta=f"{kpis['profit_margin']:.1f}%"
            )
        
        with col3:
            st.metric(
                label="🛒 Avg Order Value",
                value=f"${kpis['avg_order_value']:.2f}",
                delta="AOV"
            )
        
        with col4:
            st.metric(
                label="👥 Total Customers",
                value=f"{kpis['total_customers']:,}",
                delta="Unique"
            )
        
        with col5:
            st.metric(
                label="📊 Transactions",
                value=f"{kpis['total_transactions']:,}",
                delta="Total"
            )
        
        with col6:
            st.metric(
                label="📈 Profit Margin",
                value=f"{kpis['profit_margin']:.1f}%",
                delta="Margin"
            )
        
        # Charts Section
        st.header("📊 Analytics Dashboard")
        
        # Row 1: Revenue Trend and Category Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_revenue_trend_chart(df_filtered), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_category_performance_chart(df_filtered), use_container_width=True)
        
        # Row 2: Customer Segmentation and Regional Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_customer_segmentation_chart(df_filtered), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_regional_performance_chart(df_filtered), use_container_width=True)
        
        # Row 3: Sales Channel and Hourly Pattern
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sales_channel_chart(df_filtered), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_hourly_pattern_chart(df_filtered), use_container_width=True)
        
        # Data Table
        st.header("📋 Detailed Data")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(df_filtered.head(1000), use_container_width=True)
        
        # Download Data
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"retail_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Summary Statistics
        with st.expander("📊 Summary Statistics"):
            st.write("**Numerical Variables Summary:**")
            st.dataframe(df_filtered.describe())
            
            st.write("**Categorical Variables Summary:**")
            categorical_cols = ['product_category', 'sales_channel', 'region', 'customer_segment']
            for col in categorical_cols:
                if col in df_filtered.columns:
                    st.write(f"**{col}:**")
                    st.write(df_filtered[col].value_counts())
    
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Smart Retail Analytics Dashboard | Built with Streamlit & Plotly</p>
            <p>📊 Transforming retail data into actionable insights</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()