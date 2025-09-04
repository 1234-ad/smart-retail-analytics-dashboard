-- Retail Analytics SQL Queries
-- Collection of business intelligence queries for retail data analysis

-- =====================================================
-- 1. SALES PERFORMANCE ANALYSIS
-- =====================================================

-- Monthly revenue trend
SELECT 
    DATE_TRUNC('month', transaction_date) as month,
    SUM(total_amount) as monthly_revenue,
    COUNT(*) as transaction_count,
    AVG(total_amount) as avg_order_value
FROM sales_transactions 
GROUP BY DATE_TRUNC('month', transaction_date)
ORDER BY month;

-- Top performing products by revenue
SELECT 
    product_name,
    product_category,
    SUM(total_amount) as total_revenue,
    SUM(quantity) as units_sold,
    AVG(unit_price) as avg_price
FROM sales_transactions 
GROUP BY product_name, product_category
ORDER BY total_revenue DESC
LIMIT 20;

-- Sales channel performance comparison
SELECT 
    sales_channel,
    SUM(total_amount) as revenue,
    COUNT(*) as transactions,
    AVG(total_amount) as avg_order_value,
    SUM(total_amount) * 100.0 / SUM(SUM(total_amount)) OVER() as revenue_percentage
FROM sales_transactions 
GROUP BY sales_channel
ORDER BY revenue DESC;

-- =====================================================
-- 2. CUSTOMER ANALYTICS
-- =====================================================

-- Customer lifetime value analysis
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(total_amount) as total_spent,
        AVG(total_amount) as avg_order_value,
        MIN(transaction_date) as first_purchase,
        MAX(transaction_date) as last_purchase,
        MAX(transaction_date) - MIN(transaction_date) as customer_lifetime_days
    FROM sales_transactions 
    GROUP BY customer_id
)
SELECT 
    CASE 
        WHEN total_spent >= 1000 THEN 'High Value'
        WHEN total_spent >= 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment,
    COUNT(*) as customer_count,
    AVG(total_spent) as avg_lifetime_value,
    AVG(order_count) as avg_orders,
    AVG(customer_lifetime_days) as avg_lifetime_days
FROM customer_metrics
GROUP BY customer_segment
ORDER BY avg_lifetime_value DESC;

-- RFM Analysis (Recency, Frequency, Monetary)
WITH rfm_calc AS (
    SELECT 
        customer_id,
        CURRENT_DATE - MAX(transaction_date) as recency_days,
        COUNT(*) as frequency,
        SUM(total_amount) as monetary_value
    FROM sales_transactions 
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        recency_days,
        frequency,
        monetary_value,
        NTILE(5) OVER (ORDER BY recency_days DESC) as recency_score,
        NTILE(5) OVER (ORDER BY frequency) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value) as monetary_score
    FROM rfm_calc
)
SELECT 
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 THEN 'Potential Loyalists'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customers'
        ELSE 'Others'
    END as customer_segment,
    COUNT(*) as customer_count,
    AVG(monetary_value) as avg_monetary_value,
    AVG(frequency) as avg_frequency,
    AVG(recency_days) as avg_recency_days
FROM rfm_scores
GROUP BY customer_segment
ORDER BY avg_monetary_value DESC;

-- =====================================================
-- 3. PRODUCT PERFORMANCE ANALYSIS
-- =====================================================

-- Product category performance with growth rates
WITH monthly_category_sales AS (
    SELECT 
        product_category,
        DATE_TRUNC('month', transaction_date) as month,
        SUM(total_amount) as monthly_revenue
    FROM sales_transactions 
    GROUP BY product_category, DATE_TRUNC('month', transaction_date)
),
category_growth AS (
    SELECT 
        product_category,
        month,
        monthly_revenue,
        LAG(monthly_revenue) OVER (PARTITION BY product_category ORDER BY month) as prev_month_revenue,
        (monthly_revenue - LAG(monthly_revenue) OVER (PARTITION BY product_category ORDER BY month)) * 100.0 / 
        LAG(monthly_revenue) OVER (PARTITION BY product_category ORDER BY month) as growth_rate
    FROM monthly_category_sales
)
SELECT 
    product_category,
    AVG(monthly_revenue) as avg_monthly_revenue,
    AVG(growth_rate) as avg_growth_rate,
    STDDEV(growth_rate) as growth_volatility
FROM category_growth
WHERE prev_month_revenue IS NOT NULL
GROUP BY product_category
ORDER BY avg_monthly_revenue DESC;

-- Inventory turnover analysis
SELECT 
    product_category,
    product_name,
    SUM(quantity) as total_units_sold,
    COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as months_active,
    SUM(quantity) / COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) as avg_monthly_sales,
    SUM(total_amount) / SUM(quantity) as avg_selling_price
FROM sales_transactions 
GROUP BY product_category, product_name
HAVING COUNT(DISTINCT DATE_TRUNC('month', transaction_date)) >= 3
ORDER BY avg_monthly_sales DESC;

-- =====================================================
-- 4. SEASONAL AND TREND ANALYSIS
-- =====================================================

-- Day of week performance
SELECT 
    EXTRACT(DOW FROM transaction_date) as day_of_week,
    CASE EXTRACT(DOW FROM transaction_date)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_name,
    SUM(total_amount) as revenue,
    COUNT(*) as transactions,
    AVG(total_amount) as avg_order_value
FROM sales_transactions 
GROUP BY EXTRACT(DOW FROM transaction_date)
ORDER BY day_of_week;

-- Hourly sales pattern
SELECT 
    EXTRACT(HOUR FROM transaction_date) as hour_of_day,
    SUM(total_amount) as revenue,
    COUNT(*) as transactions,
    AVG(total_amount) as avg_order_value
FROM sales_transactions 
GROUP BY EXTRACT(HOUR FROM transaction_date)
ORDER BY hour_of_day;

-- =====================================================
-- 5. REGIONAL PERFORMANCE ANALYSIS
-- =====================================================

-- Regional sales comparison
SELECT 
    region,
    SUM(total_amount) as total_revenue,
    COUNT(*) as total_transactions,
    COUNT(DISTINCT customer_id) as unique_customers,
    AVG(total_amount) as avg_order_value,
    SUM(total_amount) / COUNT(DISTINCT customer_id) as revenue_per_customer
FROM sales_transactions 
GROUP BY region
ORDER BY total_revenue DESC;

-- Regional product preferences
WITH regional_category_sales AS (
    SELECT 
        region,
        product_category,
        SUM(total_amount) as category_revenue,
        SUM(SUM(total_amount)) OVER (PARTITION BY region) as total_regional_revenue
    FROM sales_transactions 
    GROUP BY region, product_category
)
SELECT 
    region,
    product_category,
    category_revenue,
    category_revenue * 100.0 / total_regional_revenue as percentage_of_regional_sales,
    RANK() OVER (PARTITION BY region ORDER BY category_revenue DESC) as category_rank
FROM regional_category_sales
ORDER BY region, category_rank;

-- =====================================================
-- 6. DISCOUNT AND PRICING ANALYSIS
-- =====================================================

-- Discount effectiveness analysis
SELECT 
    CASE 
        WHEN discount_percent = 0 THEN 'No Discount'
        WHEN discount_percent <= 10 THEN 'Low Discount (1-10%)'
        WHEN discount_percent <= 20 THEN 'Medium Discount (11-20%)'
        ELSE 'High Discount (>20%)'
    END as discount_tier,
    COUNT(*) as transaction_count,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order_value,
    AVG(quantity) as avg_quantity,
    AVG(discount_percent) as avg_discount_rate
FROM sales_transactions 
GROUP BY discount_tier
ORDER BY avg_order_value DESC;

-- Price elasticity analysis by category
SELECT 
    product_category,
    CASE 
        WHEN unit_price <= 50 THEN 'Low Price'
        WHEN unit_price <= 200 THEN 'Medium Price'
        ELSE 'High Price'
    END as price_tier,
    COUNT(*) as transaction_count,
    SUM(quantity) as total_quantity,
    AVG(unit_price) as avg_price,
    SUM(total_amount) as revenue
FROM sales_transactions 
GROUP BY product_category, price_tier
ORDER BY product_category, avg_price;