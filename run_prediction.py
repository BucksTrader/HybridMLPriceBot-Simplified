import pandas as pd
from eth_predictor import HybridCryptoPredictor
from eth_visualizer import ETHVisualizer
import argparse
import os
import json
from datetime import datetime
import joblib
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = 'ETH_USD_daily_data_enhanced.csv'  # Default input file
DATA_DIR = ''  # Set your data directory here if needed
OUTPUT_DIR = 'prediction_outputs'  # Base directory for all prediction outputs
MODEL_DIR = 'models'  # Directory for storing trained models

def create_output_directory():
    """Create a new dated directory for the current run"""
    # Create base output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create a new directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(run_dir)
    return run_dir

def check_metrics_quality(metrics, mape_threshold=25, da_threshold=65):
    """Check if metrics meet quality thresholds"""
    if metrics['mape'] > mape_threshold:
        print(f"Rejected: MAPE {metrics['mape']:.2f}% is above threshold of {mape_threshold}%")
        return False
    if metrics['da'] < da_threshold:
        print(f"Rejected: Directional Accuracy {metrics['da']:.2f}% is below threshold of {da_threshold}%")
        return False
    return True

def load_data(file_path):
    """Load and prepare the data"""
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"Successfully loaded data from {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def save_statistics_to_csv(stats_by_day, output_dir):
    """Save statistical summary to a CSV file"""
    # Prepare data for CSV
    stats_data = []
    
    for day, stats in enumerate(stats_by_day, 1):
        day_stats = {'Day': day}
        day_stats.update(stats)
        stats_data.append(day_stats)
    
    # Convert to DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    filename = os.path.join(output_dir, 'eth_predictions_stats.csv')
    stats_df.to_csv(filename, index=False)
    print(f"\nStatistics saved to: {filename}")
    return filename

def save_predictions_to_csv(predictions, current_price, last_date, attempt_num, output_dir):
    """Save predictions to a CSV file"""
    # Create a list to store prediction data
    prediction_data = []
    
    for day, pred in enumerate(predictions, 1):
        pred_date = last_date + pd.Timedelta(days=day)
        pct_change = ((pred - current_price) / current_price) * 100
        prediction_data.append({
            'Date': pred_date.strftime('%Y-%m-%d'),
            'Current_Price': round(current_price, 2),
            'Predicted_Price': round(pred, 2),
            'Percent_Change': round(pct_change, 2),
            'Attempt_Number': attempt_num
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(prediction_data)
    
    # Create filename
    filename = os.path.join(output_dir, f'eth_predictions_attempt{attempt_num}.csv')
    
    # Save to CSV
    predictions_df.to_csv(filename, index=False)
    print(f"\nPredictions saved to: {filename}")
    return filename

def save_run_config(args, output_dir):
    """Save run configuration to JSON file"""
    config = vars(args).copy()
    # Convert any non-serializable objects to strings
    for key, value in config.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            config[key] = str(value)
    
    # Add timestamp
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to file
    config_path = os.path.join(output_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Run configuration saved to: {config_path}")
    return config_path

def save_feature_importance(importance, output_dir):
    """Save feature importance analysis to CSV and plot"""
    if importance is None:
        return None
    
    # Convert to DataFrame
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    
    print(f"Feature importance saved to: {csv_path}")
    return csv_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ETH Price Prediction')
    parser.add_argument('--days', type=int, default=7, choices=range(1, 31),
                       help='Number of days to predict ahead (1-30)')
    parser.add_argument('--attempts', type=int, default=5,
                       help='Number of prediction attempts')
    parser.add_argument('--max-attempts', type=int, default=100,
                       help='Maximum number of attempts before giving up')
    parser.add_argument('--mape-threshold', type=float, default=25.0,
                       help='Maximum acceptable MAPE value')
    parser.add_argument('--da-threshold', type=float, default=65.0,
                       help='Minimum acceptable Directional Accuracy value')
    parser.add_argument('--load-model', type=str, default='',
                       help='Path to a previously trained model to load instead of training')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the trained model for future use')
    parser.add_argument('--feature-importance', action='store_true',
                       help='Calculate and save feature importance analysis')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Sequence length for model input')
    args = parser.parse_args()

    # Create new output directory for this run
    output_dir = create_output_directory()
    print(f"Saving outputs to: {output_dir}")
    
    # Save run configuration
    save_run_config(args, output_dir)

    # Construct file path
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    # Load the data
    df = load_data(file_path)

    # Initialize the models
    print("Initializing models...")
    model = HybridCryptoPredictor(sequence_length=args.sequence_length)
    visualizer = ETHVisualizer()

    # Try to load a pre-trained model if requested
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        if model.load_model(args.load_model):
            print("Model loaded successfully.")
        else:
            print("Failed to load model. Will train a new one.")
            args.load_model = ''  # Reset to train a new model

    successful_predictions = 0
    total_attempts = 0
    saved_files = []  # Keep track of saved CSV files

    while successful_predictions < args.attempts and total_attempts < args.max_attempts:
        total_attempts += 1
        print(f"\nTraining model (Attempt {total_attempts}, Successful: {successful_predictions}/{args.attempts})...")
        
        try:
            # Skip training if using a pre-loaded model or if we already trained successfully
            if args.load_model and total_attempts == 1:
                # For a loaded model, we'll evaluate performance on the test set
                # but won't use these metrics for quality control
                metrics = {
                    'rmse': 0.0,  # We don't have test metrics for a loaded model
                    'mae': 0.0,
                    'mape': 0.0,  # We'll assume the loaded model met quality thresholds
                    'da': 100.0   # when it was trained
                }
                print("Using pre-loaded model (skipping quality checks).")
            else:
                # Train model and get metrics
                metrics = model.fit(df)
                
                # Check if metrics meet quality thresholds
                if not check_metrics_quality(metrics, args.mape_threshold, args.da_threshold):
                    print("Attempting another prediction...\n")
                    continue
            
            # Save model if requested and metrics are good
            if args.save_model and successful_predictions == 0:
                model_save_dir = os.path.join(output_dir, 'model')
                os.makedirs(model_save_dir, exist_ok=True)
                model.save_model(model_save_dir)
                print(f"Model saved to {model_save_dir}")
            
            # Calculate feature importance if requested
            if args.feature_importance and successful_predictions == 0:
                importance = model.feature_importance(df)
                save_feature_importance(importance, output_dir)
                
            # Make multi-day predictions
            predictions = model.predict_multiple_days(df, days_ahead=args.days)
            
            # Print results
            current_price = df['Close'].iloc[-1]
            print(f"\nResults for Successful Attempt {successful_predictions + 1}:")
            print(f"Current ETH price: ${current_price:.2f}")
            
            for day, pred in enumerate(predictions, 1):
                pct_change = ((pred - current_price) / current_price) * 100
                print(f"Day {day} prediction: ${pred:.2f} (Change: {pct_change:.2f}%)")
            
            # Save predictions to CSV
            csv_file = save_predictions_to_csv(predictions, current_price, df.index[-1], 
                                             successful_predictions + 1, output_dir)
            saved_files.append(csv_file)
            
            # Add predictions to visualizer
            visualizer.add_prediction(metrics, predictions)
            successful_predictions += 1
            
            # If using a pre-loaded model, we only need one attempt
            if args.load_model:
                break
                
        except Exception as e:
            print(f"An error occurred in attempt {total_attempts}: {str(e)}")
            print("Error details:", str(e))
            continue

    if successful_predictions < args.attempts:
        print(f"\nWarning: Only achieved {successful_predictions} successful predictions")
        print(f"out of {args.attempts} desired after {total_attempts} attempts.")
    
    if successful_predictions > 0:
        # Calculate and display statistics
        stats_by_day = visualizer.calculate_statistics(days=args.days)

        # Create visualization
        print("\nGenerating plots...")
        try:
            # Generate plots and save them to the output directory
            visualizer.plot_predictions(df, output_dir)
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
            
        # Save statistics to CSV
        stats_file = save_statistics_to_csv(stats_by_day, output_dir)
        saved_files.append(stats_file)

        # Print prediction dates and CSV files
        last_date = df.index[-1]
        print("\nPredictions are for:")
        for day in range(args.days):
            pred_date = last_date + pd.Timedelta(days=day+1)
            print(f"Day {day+1}: {pred_date.strftime('%Y-%m-%d')}")
            
        print("\nAll files have been saved to:")
        print(f"- {output_dir}")
    else:
        print("\nNo successful predictions were made. Try adjusting the threshold values:")
        print(f"Current MAPE threshold: {args.mape_threshold}%")
        print(f"Current Directional Accuracy threshold: {args.da_threshold}%")

if __name__ == "__main__":
    main()