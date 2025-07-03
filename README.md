# HybridMLPriceBot-Simplified

**HybridMLPriceBot-Simplified** is a Python-based Ethereum price prediction system that leverages a hybrid LSTM-CNN deep learning model, technical indicators, and robust visualization tools. This repository provides a streamlined, reproducible workflow for training, evaluating, and visualizing ETH price forecasts.

## Features

- **Hybrid LSTM-CNN Model**: Combines LSTM and CNN layers for advanced time series forecasting.
- **Rich Technical Indicators**: Incorporates a wide range of features (moving averages, momentum, volatility, volume, Ichimoku, money flow, etc.).
- **Automated Training & Evaluation**: Includes early stopping, learning rate scheduling, and quality control (MAPE, DA thresholds).
- **Multi-day Prediction**: Predicts ETH prices for up to 30 days ahead.
- **Comprehensive Visualization**: Generates multi-panel plots of price, indicators, and prediction statistics.
- **Model Persistence**: Save and load trained models and scalers for reproducibility.

## File Overview

- **`eth_predictor.py`**  
  Contains the `HybridCryptoPredictor` class, which:
  - Prepares features and sequences for model training.
  - Defines and trains the LSTM-CNN model.
  - Evaluates performance (RMSE, MAE, MAPE, Directional Accuracy).
  - Supports model saving/loading and feature importance analysis.

- **`eth_visualizer.py`**  
  Contains the `ETHVisualizer` class, which:
  - Plots ETH price with technical indicators and predictions.
  - Provides statistical summaries and histograms of predictions.
  - Visualizes feature importance.

- **`run_prediction.py`**  
  Command-line script to:
  - Train or load a model.
  - Run multi-day predictions.
  - Save results and statistics to CSV.
  - Generate and save visualizations.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
*(You may need to create this file based on your environment: pandas, numpy, scikit-learn, tensorflow, matplotlib, etc.)*

### 2. Prepare Data

- Place your ETH price data (with technical indicators) in `ETH_USD_daily_data_enhanced.csv`.

### 3. Train and Predict

```bash
python run_prediction.py --days 7 --attempts 5 --save-model
```

- Adjust `--days` for prediction horizon, `--attempts` for number of runs, and `--save-model` to persist the trained model.

### 4. Visualize Results

- Plots and CSVs will be saved in the `prediction_outputs/` directory.

## Example Output

- **Prediction statistics**: RMSE, MAE, MAPE, Directional Accuracy
- **Plots**: ETH price with technical indicators, prediction distributions, feature importance

## Customization

- **Model architecture**: Edit `build_model()` in `eth_predictor.py`.
- **Indicators/features**: Modify the `self.columns` list in `HybridCryptoPredictor`.
- **Quality thresholds**: Adjust `mape_threshold` and `da_threshold` in `run_prediction.py`.

## License

MIT License

---

## Acknowledgments

- Built with TensorFlow, scikit-learn, pandas, and matplotlib.
- Inspired by best practices in time series forecasting and quantitative finance.
