"""
Sales Forecasting Model
======================

This module implements various forecasting models for retail sales prediction
including ARIMA, Prophet, and machine learning approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")


class SalesForecaster:
    """
    Comprehensive sales forecasting class with multiple modeling approaches.
    """
    
    def __init__(self, data, date_col='date', target_col='sales'):
        """
        Initialize the forecaster with sales data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Sales data with date and target columns
        date_col : str
            Name of the date column
        target_col : str
            Name of the target variable (sales)
        """
        self.data = data.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.models = {}
        self.forecasts = {}
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and clean the data for modeling."""
        # Ensure date column is datetime
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        
        # Sort by date
        self.data = self.data.sort_values(self.date_col).reset_index(drop=True)
        
        # Create time series
        self.ts_data = self.data.set_index(self.date_col)[self.target_col]
        
        # Remove any missing values
        self.ts_data = self.ts_data.dropna()
        
        print(f"Data prepared: {len(self.ts_data)} observations from {self.ts_data.index.min()} to {self.ts_data.index.max()}")
    
    def check_stationarity(self, plot=True):
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        """
        result = adfuller(self.ts_data.dropna())
        
        print('Augmented Dickey-Fuller Test Results:')
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("Series is stationary (reject null hypothesis)")
            is_stationary = True
        else:
            print("Series is non-stationary (fail to reject null hypothesis)")
            is_stationary = False
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Original series
            axes[0,0].plot(self.ts_data)
            axes[0,0].set_title('Original Time Series')
            axes[0,0].set_ylabel('Sales')
            
            # Rolling statistics
            rolling_mean = self.ts_data.rolling(window=30).mean()
            rolling_std = self.ts_data.rolling(window=30).std()
            
            axes[0,1].plot(self.ts_data, label='Original')
            axes[0,1].plot(rolling_mean, label='Rolling Mean')
            axes[0,1].plot(rolling_std, label='Rolling Std')
            axes[0,1].set_title('Rolling Statistics')
            axes[0,1].legend()
            
            # ACF and PACF
            plot_acf(self.ts_data.dropna(), ax=axes[1,0], lags=40)
            plot_pacf(self.ts_data.dropna(), ax=axes[1,1], lags=40)
            
            plt.tight_layout()
            plt.show()
        
        return is_stationary
    
    def decompose_series(self, model='additive', period=365):
        """
        Decompose the time series into trend, seasonal, and residual components.
        """
        decomposition = seasonal_decompose(self.ts_data, model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def fit_arima(self, order=(1,1,1), seasonal_order=None):
        """
        Fit ARIMA model to the time series.
        """
        try:
            if seasonal_order:
                model = ARIMA(self.ts_data, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(self.ts_data, order=order)
            
            fitted_model = model.fit()
            self.models['arima'] = fitted_model
            
            print("ARIMA Model Summary:")
            print(fitted_model.summary())
            
            return fitted_model
        
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    
    def fit_prophet(self):
        """
        Fit Prophet model to the time series.
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Please install it first.")
            return None
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': self.ts_data.index,
            'y': self.ts_data.values
        })
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_data)
        self.models['prophet'] = model
        
        print("Prophet model fitted successfully")
        return model
    
    def create_features(self, df):
        """
        Create time-based features for machine learning models.
        """
        df = df.copy()
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['day'] = df[self.date_col].dt.day
        df['dayofweek'] = df[self.date_col].dt.dayofweek
        df['quarter'] = df[self.date_col].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'sales_lag_{lag}'] = df[self.target_col].shift(lag)
        
        # Rolling features
        for window in [7, 30]:
            df[f'sales_rolling_mean_{window}'] = df[self.target_col].rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df[self.target_col].rolling(window=window).std()
        
        return df
    
    def fit_ml_models(self):
        """
        Fit machine learning models for forecasting.
        """
        # Create features
        ml_data = self.create_features(self.data)
        
        # Remove rows with NaN values (due to lag and rolling features)
        ml_data = ml_data.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in ml_data.columns if col not in [self.date_col, self.target_col]]
        X = ml_data[feature_cols]
        y = ml_data[self.target_col]
        
        # Split data (time series split)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Models to try
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features for linear regression
                if name == 'linear_regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                scores.append(mae)
            
            avg_mae = np.mean(scores)
            results[name] = {'model': model, 'mae': avg_mae}
            print(f"{name}: Average MAE = {avg_mae:.2f}")
        
        # Store best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        self.models['ml_best'] = results[best_model_name]['model']
        
        print(f"Best ML model: {best_model_name}")
        return results
    
    def forecast(self, periods=30, model_type='all'):
        """
        Generate forecasts using specified models.
        """
        forecasts = {}
        
        if model_type in ['all', 'arima'] and 'arima' in self.models:
            # ARIMA forecast
            arima_forecast = self.models['arima'].forecast(steps=periods)
            forecasts['arima'] = arima_forecast
        
        if model_type in ['all', 'prophet'] and 'prophet' in self.models:
            # Prophet forecast
            future_dates = self.models['prophet'].make_future_dataframe(periods=periods)
            prophet_forecast = self.models['prophet'].predict(future_dates)
            forecasts['prophet'] = prophet_forecast['yhat'].tail(periods).values
        
        self.forecasts = forecasts
        return forecasts
    
    def plot_forecasts(self, periods=30):
        """
        Plot historical data and forecasts.
        """
        if not self.forecasts:
            print("No forecasts available. Run forecast() first.")
            return
        
        # Create future dates
        last_date = self.ts_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(self.ts_data.index[-90:], self.ts_data.values[-90:], label='Historical', color='black', linewidth=2)
        
        # Plot forecasts
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, forecast) in enumerate(self.forecasts.items()):
            plt.plot(future_dates, forecast, label=f'{model_name.upper()} Forecast', 
                    color=colors[i % len(colors)], linestyle='--', linewidth=2)
        
        plt.title('Sales Forecasting Results')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def evaluate_models(self, test_size=0.2):
        """
        Evaluate model performance on test data.
        """
        # Split data
        split_point = int(len(self.ts_data) * (1 - test_size))
        train_data = self.ts_data[:split_point]
        test_data = self.ts_data[split_point:]
        
        results = {}
        
        # Evaluate each model
        for model_name in self.models.keys():
            if model_name == 'arima':
                # Refit ARIMA on training data
                model = ARIMA(train_data, order=(1,1,1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=len(test_data))
                
            elif model_name == 'prophet' and PROPHET_AVAILABLE:
                # Refit Prophet on training data
                prophet_train = pd.DataFrame({
                    'ds': train_data.index,
                    'y': train_data.values
                })
                model = Prophet()
                model.fit(prophet_train)
                
                future = model.make_future_dataframe(periods=len(test_data))
                forecast_df = model.predict(future)
                forecast = forecast_df['yhat'].tail(len(test_data)).values
            
            else:
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            results[model_name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }
        
        # Display results
        results_df = pd.DataFrame(results).T
        print("Model Evaluation Results:")
        print(results_df.round(2))
        
        return results_df


# Example usage and utility functions
def generate_sample_sales_data(start_date='2022-01-01', periods=365):
    """
    Generate sample sales data for testing.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Create trend and seasonality
    trend = np.linspace(1000, 1500, periods)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    weekly = 100 * np.sin(2 * np.pi * np.arange(periods) / 7)
    noise = np.random.normal(0, 50, periods)
    
    sales = trend + seasonal + weekly + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative sales
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales
    })


if __name__ == "__main__":
    # Example usage
    print("Sales Forecasting Model Example")
    print("=" * 40)
    
    # Generate sample data
    sample_data = generate_sample_sales_data()
    print(f"Generated {len(sample_data)} days of sample sales data")
    
    # Initialize forecaster
    forecaster = SalesForecaster(sample_data, date_col='date', target_col='sales')
    
    # Check stationarity
    forecaster.check_stationarity(plot=False)
    
    # Fit models
    print("\nFitting ARIMA model...")
    forecaster.fit_arima(order=(1,1,1))
    
    if PROPHET_AVAILABLE:
        print("\nFitting Prophet model...")
        forecaster.fit_prophet()
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = forecaster.forecast(periods=30)
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluation = forecaster.evaluate_models()
    
    print("\nForecasting complete!")