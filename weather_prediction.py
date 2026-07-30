"""
Weather Data Analysis and Prediction
Analyzes historical weather data and predicts future temperature trends.
Uses Linear Regression and time series analysis techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    """A class to analyze and predict weather patterns."""
    
    def __init__(self):
        self.model = LinearRegression()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath='weather_data.csv'):
        """Load weather data from CSV file."""
        try:
            self.df = pd.read_csv(filepath)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"✓ Data loaded successfully: {len(self.df)} records")
            print(f"  - Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
            print(f"  - Duration: {(self.df['date'].max() - self.df['date'].min()).days} days")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: {filepath} not found!")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\n📊 Temperature Statistics:")
        print(f"  - Mean temperature: {self.df['temperature'].mean():.2f}°C")
        print(f"  - Min temperature: {self.df['temperature'].min():.2f}°C")
        print(f"  - Max temperature: {self.df['temperature'].max():.2f}°C")
        print(f"  - Standard deviation: {self.df['temperature'].std():.2f}°C")
        
        if 'humidity' in self.df.columns:
            print("\n💧 Humidity Statistics:")
            print(f"  - Mean humidity: {self.df['humidity'].mean():.2f}%")
            print(f"  - Min humidity: {self.df['humidity'].min():.2f}%")
            print(f"  - Max humidity: {self.df['humidity'].max():.2f}%")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print("\n⚠️ Missing Values:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values detected")
    
    def visualize_trends(self):
        """Visualize temperature trends over time."""
        print("\n📈 Generating visualizations...")
        
        # Set up the plot style
        sns.set_style("whitegrid")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weather Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temperature over time
        axes[0, 0].plot(self.df['date'], self.df['temperature'], color='orangered', linewidth=1)
        axes[0, 0].set_title('Temperature Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Temperature distribution
        axes[0, 1].hist(self.df['temperature'], bins=30, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Temperature Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly average temperature
        self.df['month'] = self.df['date'].dt.month
        monthly_avg = self.df.groupby('month')['temperature'].mean()
        axes[1, 0].bar(monthly_avg.index, monthly_avg.values, color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_title('Average Temperature by Month', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Temperature (°C)')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Temperature vs Humidity (if available)
        if 'humidity' in self.df.columns:
            axes[1, 1].scatter(self.df['humidity'], self.df['temperature'], 
                              alpha=0.5, color='purple', s=20)
            axes[1, 1].set_title('Temperature vs Humidity', fontweight='bold')
            axes[1, 1].set_xlabel('Humidity (%)')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Box plot by season if humidity not available
            self.df['season'] = self.df['month'].apply(self._get_season)
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            axes[1, 1].boxplot([self.df[self.df['season'] == s]['temperature'].values 
                                for s in season_order],
                               labels=season_order)
            axes[1, 1].set_title('Temperature by Season', fontweight='bold')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weather_analysis.png', dpi=100, bbox_inches='tight')
        print("✓ Visualization saved as 'weather_analysis.png'")
        plt.close()
    
    def _get_season(self, month):
        """Helper function to determine season from month."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def prepare_features(self):
        """Prepare features for machine learning model."""
        print("\n🔧 Preparing features for prediction...")
        
        # Create time-based features
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        
        # Create cyclical features (sine and cosine) to capture seasonality
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        
        # Select features
        feature_cols = ['day_of_year', 'year', 'month', 'day_sin', 'day_cos']
        
        # Add humidity if available
        if 'humidity' in self.df.columns:
            feature_cols.append('humidity')
        
        X = self.df[feature_cols]
        y = self.df['temperature']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Features prepared successfully")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        print(f"  - Features used: {', '.join(feature_cols)}")
        
        return self.X_train, self.X_test
    
    def train_model(self):
        """Train the Linear Regression model."""
        print("\n🔄 Training Linear Regression model...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained successfully")
    
    def evaluate_model(self):
        """Evaluate model performance."""
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        
        print("\n📊 Training Set Performance:")
        print(f"  - MAE (Mean Absolute Error): {train_mae:.4f}°C")
        print(f"  - RMSE (Root Mean Squared Error): {train_rmse:.4f}°C")
        print(f"  - R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")
        
        print("\n🎯 Testing Set Performance:")
        print(f"  - MAE (Mean Absolute Error): {test_mae:.4f}°C")
        print(f"  - RMSE (Root Mean Squared Error): {test_rmse:.4f}°C")
        print(f"  - R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
        
        # Interpretation
        print("\n💡 Interpretation:")
        print(f"  - On average, predictions are off by {test_mae:.2f}°C")
        print(f"  - Model explains {test_r2*100:.2f}% of temperature variance")
        
        # Visualize predictions
        self._plot_predictions(y_test_pred)
        
        return test_mae, test_rmse, test_r2
    
    def _plot_predictions(self, y_pred):
        """Plot actual vs predicted temperatures."""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Actual vs Predicted scatter
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.5, color='blue', s=20)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Temperature (°C)', fontweight='bold')
        plt.ylabel('Predicted Temperature (°C)', fontweight='bold')
        plt.title('Actual vs Predicted Temperature', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Prediction errors
        plt.subplot(1, 2, 2)
        errors = self.y_test - y_pred
        plt.hist(errors, bins=30, color='green', edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (°C)', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Distribution of Prediction Errors', fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=100, bbox_inches='tight')
        print("✓ Prediction visualization saved as 'prediction_results.png'")
        plt.close()
    
    def predict_future(self, days_ahead=30):
        """Predict temperature for future days."""
        print(f"\n🔮 Predicting temperature for next {days_ahead} days...")
        
        # Get the last date in dataset
        last_date = self.df['date'].max()
        
        # Create future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create features for future dates
        future_data = pd.DataFrame({
            'date': future_dates,
            'day_of_year': [d.timetuple().tm_yday for d in future_dates],
            'year': [d.year for d in future_dates],
            'month': [d.month for d in future_dates],
        })
        
        # Add cyclical features
        future_data['day_sin'] = np.sin(2 * np.pi * future_data['day_of_year'] / 365)
        future_data['day_cos'] = np.cos(2 * np.pi * future_data['day_of_year'] / 365)
        
        # Add humidity if it was used in training
        if 'humidity' in self.X_train.columns:
            # Use average humidity from training data
            avg_humidity = self.df['humidity'].mean()
            future_data['humidity'] = avg_humidity
            print(f"  Note: Using average humidity ({avg_humidity:.1f}%) for predictions")
        
        # Select features in the same order as training
        X_future = future_data[self.X_train.columns]
        
        # Make predictions
        future_temps = self.model.predict(X_future)
        
        # Display predictions
        print("\n🌡️ Temperature Forecast:")
        print("="*60)
        for i in range(min(10, days_ahead)):  # Show first 10 days
            date_str = future_dates[i].strftime('%Y-%m-%d')
            temp = future_temps[i]
            print(f"  {date_str}: {temp:.2f}°C")
        
        if days_ahead > 10:
            print(f"  ... ({days_ahead - 10} more days)")
        
        # Visualize future predictions
        self._plot_future_predictions(future_dates, future_temps)
        
        return future_dates, future_temps
    
    def _plot_future_predictions(self, future_dates, future_temps):
        """Plot historical data and future predictions."""
        plt.figure(figsize=(14, 6))
        
        # Plot historical data
        plt.plot(self.df['date'], self.df['temperature'], 
                label='Historical Data', color='blue', linewidth=1.5)
        
        # Plot future predictions
        plt.plot(future_dates, future_temps, 
                label='Future Predictions', color='red', linewidth=2, linestyle='--')
        
        # Add vertical line at prediction start
        plt.axvline(x=self.df['date'].max(), color='green', 
                   linestyle=':', linewidth=2, label='Prediction Start')
        
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Temperature (°C)', fontweight='bold')
        plt.title('Historical Temperature and Future Predictions', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('future_predictions.png', dpi=100, bbox_inches='tight')
        print("✓ Future predictions visualization saved as 'future_predictions.png'")
        plt.close()

def main():
    """Main function to run weather analysis and prediction."""
    print("="*60)
    print("    WEATHER DATA ANALYSIS & PREDICTION    ")
    print("="*60)
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Load data
    df = predictor.load_data('weather_data.csv')
    if df is None:
        return
    
    # Explore data
    predictor.explore_data()
    
    # Visualize trends
    predictor.visualize_trends()
    
    # Prepare features
    predictor.prepare_features()
    
    # Train model
    predictor.train_model()
    
    # Evaluate model
    mae, rmse, r2 = predictor.evaluate_model()
    
    # Predict future
    future_dates, future_temps = predictor.predict_future(days_ahead=30)
    
    print("\n" + "="*60)
    print("✓ Weather analysis and prediction completed successfully!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("  - weather_analysis.png (EDA visualizations)")
    print("  - prediction_results.png (Model performance)")
    print("  - future_predictions.png (Future temperature forecast)")
    print("\n💡 Summary:")
    print(f"  - Model Accuracy (R²): {r2*100:.2f}%")
    print(f"  - Average Prediction Error: ±{mae:.2f}°C")
    print(f"  - Next 30 days predicted successfully")
    print("="*60)

if __name__ == "__main__":
    main()