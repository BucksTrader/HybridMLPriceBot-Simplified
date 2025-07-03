import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from pathlib import Path

class ETHVisualizer:
    def __init__(self):
        self.predictions = []
        
    def add_prediction(self, metrics, predicted_prices):
        """Add a new prediction and its metrics"""
        self.predictions.append({
            'RMSE': metrics['rmse'] if isinstance(metrics, dict) and 'rmse' in metrics else 0,
            'MAE': metrics['mae'] if isinstance(metrics, dict) and 'mae' in metrics else 0,
            'MAPE': metrics['mape'] if isinstance(metrics, dict) and 'mape' in metrics else 0,
            'DA': metrics['da'] if isinstance(metrics, dict) and 'da' in metrics else 0,
            'Predicted Prices': predicted_prices if isinstance(predicted_prices, np.ndarray) 
                              else np.array(predicted_prices)
        })
    
    def calculate_statistics(self, days=1):
        """Calculate statistics for predictions"""
        print("\nPrediction Statistics:")
        
        stats_by_day = []
        for day in range(days):
            day_predictions = [pred['Predicted Prices'][day] if day < len(pred['Predicted Prices']) 
                              else None for pred in self.predictions]
            day_predictions = [p for p in day_predictions if p is not None]
            
            if not day_predictions:
                continue
            
            stats_dict = {
                'Mean': np.mean(day_predictions),
                'Median': np.median(day_predictions),
                'Mode': float(stats.mode(day_predictions, axis=None)[0]),
                'Range': np.ptp(day_predictions),
                'Std Dev': np.std(day_predictions),
                'Skewness': stats.skew(day_predictions),
                'Kurtosis': stats.kurtosis(day_predictions)
            }
            
            print(f"\nDay {day + 1} Statistics:")
            for stat_name, value in stats_dict.items():
                print(f"{stat_name}: ${value:.2f}" if 'Range' in stat_name or 'Mean' in stat_name or 'Median' in stat_name or 'Mode' in stat_name
                      else f"{stat_name}: {value:.4f}")
            
            stats_by_day.append(stats_dict)
            
        return stats_by_day
    
    def plot_predictions(self, df, output_dir=None):
        """Create comprehensive price plots with synchronized technical indicators"""
        # Create figure
        fig = plt.figure(figsize=(15, 24))
        
        # Create grid layout
        gs = fig.add_gridspec(5, 2, height_ratios=[3, 2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Create all subplots with shared x-axis
        ax_price = fig.add_subplot(gs[0, :])  # Price chart spans both columns
        ax_ichimoku = fig.add_subplot(gs[1, :], sharex=ax_price)  # Ichimoku spans both columns
        
        # Create other indicators in 2x3 grid below
        ax_macd = fig.add_subplot(gs[2, 0], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2, 1], sharex=ax_price)
        ax_volume = fig.add_subplot(gs[3, 0], sharex=ax_price)
        ax_momentum = fig.add_subplot(gs[3, 1], sharex=ax_price)
        ax_volatility = fig.add_subplot(gs[4, 0], sharex=ax_price)
        ax_money_flow = fig.add_subplot(gs[4, 1], sharex=ax_price)

        # Plot on each subplot
        self._plot_main_price_chart(df, ax_price)
        self._plot_ichimoku_cloud(df, ax_ichimoku)
        self._plot_macd(df, ax_macd)
        self._plot_rsi_williams(df, ax_rsi)
        self._plot_volume_analysis(df, ax_volume)
        self._plot_momentum_indicators(df, ax_momentum)
        self._plot_volatility_indicators(df, ax_volatility)
        self._plot_money_flow(df, ax_money_flow)

        # Hide x-axis labels except for bottom plots
        for ax in [ax_price, ax_ichimoku, ax_macd, ax_rsi]:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        # Set proper date formatting for x-axis on bottom plots
        for ax in [ax_volume, ax_momentum, ax_volatility, ax_money_flow]:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add title with timestamp
        plt.suptitle(f'ETH Price Analysis - {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=16, y=0.995)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'eth_technical_analysis.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'eth_technical_analysis.pdf'), bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    def plot_focused_prediction(self, df, output_dir=None):
        """Plot a focused view of the predictions with recent price history"""
        if len(self.predictions) == 0:
            print("No predictions available to plot.")
            return
            
        # Setup the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot recent history (last 30 days)
        history_days = 30
        recent_df = df.iloc[-history_days:]
        ax.plot(recent_df.index, recent_df['Close'], label='Historical ETH Price', 
                color='black', linewidth=2)
        
        # Plot predictions
        last_date = df.index[-1]
        
        # Color palette for multiple predictions
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.predictions)))
        
        for i, prediction in enumerate(self.predictions):
            pred_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=len(prediction['Predicted Prices']), 
                freq='D'
            )
            
            ax.plot(pred_dates, prediction['Predicted Prices'], 
                   marker='o', linestyle='-', 
                   label=f"Prediction {i+1} (MAPE: {prediction['MAPE']:.1f}%, DA: {prediction['DA']:.1f}%)",
                   color=colors[i])
        
        # Calculate mean prediction if we have multiple predictions
        if len(self.predictions) > 1:
            max_days = max(len(p['Predicted Prices']) for p in self.predictions)
            mean_preds = []
            
            for day in range(max_days):
                day_values = [p['Predicted Prices'][day] for p in self.predictions 
                             if day < len(p['Predicted Prices'])]
                if day_values:
                    mean_preds.append(np.mean(day_values))
            
            if mean_preds:
                mean_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(mean_preds),
                    freq='D'
                )
                ax.plot(mean_dates, mean_preds, 'r--', linewidth=3, 
                       label='Mean Prediction')
        
        # Add labels and formatting
        ax.set_title('ETH Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='best')
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'eth_focused_prediction.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'eth_focused_prediction.pdf'), bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    def plot_prediction_histogram(self, day=0, output_dir=None):
        """Plot histogram of predictions for a specific day ahead"""
        if len(self.predictions) < 2:
            print("Need at least 2 predictions to create a histogram.")
            return
            
        # Extract predictions for the specified day
        day_predictions = [p['Predicted Prices'][day] for p in self.predictions
                          if day < len(p['Predicted Prices'])]
        
        if not day_predictions:
            print(f"No predictions available for day {day+1}.")
            return
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(day_predictions, bins=10, alpha=0.7, color='blue')
        
        # Add mean, median lines
        mean_val = np.mean(day_predictions)
        median_val = np.median(day_predictions)
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_val:.2f}')
        
        # Add labels
        ax.set_title(f'Distribution of Day {day+1} Predictions')
        ax.set_xlabel('Predicted Price (USD)')
        ax.set_ylabel('Frequency')
        
        # Add legend
        ax.legend()
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'eth_prediction_histogram_day{day+1}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'eth_prediction_histogram_day{day+1}.pdf'), bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    def plot_feature_importance(self, importance_dict, output_dir=None):
        """Plot feature importance analysis"""
        if not importance_dict:
            print("No feature importance data provided.")
            return
            
        # Extract data
        features = list(importance_dict.keys())
        importance_values = list(importance_dict.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)
        features = [features[i] for i in sorted_indices]
        importance_values = [importance_values[i] for i in sorted_indices]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        
        # Labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance Analysis')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'), bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    def _plot_main_price_chart(self, df, ax):
        """Plot main price chart with multiple indicators"""
        # Price and moving averages
        ax.plot(df.index, df['Close'], label='ETH Price', color='black', linewidth=2)
        ax.plot(df['SMA_20'], label='20-day MA', linestyle='--', color='blue', alpha=0.6)
        ax.plot(df['EMA_20'], label='20-day EMA', linestyle='--', color='red', alpha=0.6)
        ax.plot(df['KAMA'], label='KAMA', linestyle='--', color='purple', alpha=0.6)
        
        # Bollinger Bands
        ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
        ax.plot(df['BB_Upper'], label='BB Upper', color='gray', linestyle=':')
        ax.plot(df['BB_Lower'], label='BB Lower', color='gray', linestyle=':')
        
        # Donchian Channels
        ax.plot(df['Donchian_High'], label='Donchian High', color='green', linestyle=':')
        ax.plot(df['Donchian_Low'], label='Donchian Low', color='red', linestyle=':')
        
        # Plot predictions
        last_date = df.index[-1]
        
        # Define colors for predictions
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.predictions)))
        
        for idx, prediction in enumerate(self.predictions):
            prediction_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=len(prediction['Predicted Prices']), 
                freq='D'
            )
            ax.plot(prediction_dates, prediction['Predicted Prices'], 
                   marker='o', linestyle='-', label=f'Prediction {idx+1}',
                   color=colors[idx])
        
        ax.set_ylabel('ETH Price (USD)')
        ax.set_title('ETH Price with Technical Indicators and Predictions')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        
    def _plot_ichimoku_cloud(self, df, ax):
        """Plot Ichimoku Cloud"""
        ax.plot(df.index, df['Ichimoku_Conversion'], label='Conversion Line', color='blue')
        ax.plot(df.index, df['Ichimoku_Base'], label='Base Line', color='red')
        ax.fill_between(df.index, df['Ichimoku_A'], df['Ichimoku_B'], 
                       where=df['Ichimoku_A'] >= df['Ichimoku_B'],
                       color='green', alpha=0.1)
        ax.fill_between(df.index, df['Ichimoku_A'], df['Ichimoku_B'],
                       where=df['Ichimoku_A'] < df['Ichimoku_B'],
                       color='red', alpha=0.1)
        ax.set_ylabel('Price')
        ax.set_title('Ichimoku Cloud')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        
    def _plot_macd(self, df, ax):
        """Plot MACD"""
        ax.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
        ax.bar(df.index, df['MACD_Hist'], label='Histogram', color='gray', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax.set_ylabel('MACD')
        ax.set_title('MACD')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        
    def _plot_rsi_williams(self, df, ax):
        """Plot RSI and Williams %R"""
        ax1 = ax
        ax2 = ax1.twinx()
        
        # RSI
        ax1.plot(df.index, df['RSI_14'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax1.set_ylabel('RSI', color='purple')
        
        # Williams %R
        ax2.plot(df.index, df['Williams_R'], label='Williams %R', color='orange')
        ax2.set_ylabel('Williams %R', color='orange')
        
        ax1.set_title('RSI and Williams %R')
        ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.3))
        
    def _plot_volume_analysis(self, df, ax):
        """Plot volume analysis"""
        ax1 = ax
        ax2 = ax1.twinx()
        
        # Volume bars
        ax1.bar(df.index, df['Volume'], label='Volume', color='gray', alpha=0.3)
        ax1.plot(df.index, df['Volume_MA_20'], label='Volume MA(20)', color='blue')
        ax1.set_ylabel('Volume')
        
        # Force Index
        ax2.plot(df.index, df['Force_Index'], label='Force Index', color='red')
        ax2.set_ylabel('Force Index', color='red')
        
        ax1.set_title('Volume Analysis')
        ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.7))
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.3))
        
    def _plot_momentum_indicators(self, df, ax):
        """Plot momentum indicators"""
        ax.plot(df.index, df['Price_ROC_10'], label='ROC(10)', color='blue')
        ax.plot(df.index, df['PMO_Line'], label='PMO', color='red')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax.set_ylabel('Momentum')
        ax.set_title('Momentum Indicators')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        
    def _plot_volatility_indicators(self, df, ax):
        """Plot volatility indicators"""
        ax.plot(df.index, df['ATR'], label='ATR', color='blue')
        ax.plot(df.index, df['Volatility_Ratio'], label='Volatility Ratio', color='red')
        ax.set_ylabel('Volatility')
        ax.set_title('Volatility Indicators')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        
    def _plot_money_flow(self, df, ax):
        """Plot money flow indicators"""
        ax.plot(df.index, df['Money_Flow_Pos'].rolling(20).mean(), 
                label='Positive Money Flow', color='green')
        ax.plot(df.index, df['Money_Flow_Neg'].rolling(20).mean(),
                label='Negative Money Flow', color='red')
        ax.set_ylabel('Money Flow')
        ax.set_title('Money Flow Analysis')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))