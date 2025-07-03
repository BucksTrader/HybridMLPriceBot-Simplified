import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import math
import warnings
import os
import joblib
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class HybridCryptoPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.columns = [
            # Price data
            'Close', 'Open', 'High', 'Low',
            
            # Moving Averages
            'SMA_20', 'Distance_From_SMA_20',
            'EMA_20', 'Distance_From_EMA_20',
            'KAMA',
            
            # Momentum Indicators
            'RSI_14', 'Williams_R',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Price_ROC_5', 'Price_ROC_10', 'Price_ROC_20',
            'PMO_Line',
            
            # Volatility Indicators
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width',
            'ATR', 'ATR_Pct', 'Volatility_Ratio',
            'Donchian_High', 'Donchian_Low', 'Donchian_Middle',
            
            # Volume Indicators
            'Volume', 'Volume_MA_20', 'Volume_Ratio',
            'Force_Index', 'EOM',
            'Volume_ROC_5', 'Volume_ROC_10',
            
            # Ichimoku Components
            'Ichimoku_Conversion', 'Ichimoku_Base',
            'Ichimoku_A', 'Ichimoku_B',
            
            # Money Flow Indicators
            'Money_Flow', 'Money_Flow_Pos', 'Money_Flow_Neg',
            
            # Market Structure
            'Price_Position', 'Volume_Position',
            'Gap', 'Gap_Percent'
        ]
        
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM"""
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        """Calculate MAPE"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    def directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy"""
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
        accuracy = np.sum(actual_direction == predicted_direction) / len(actual_direction)
        return accuracy * 100
    
    def build_model(self, input_shape):
        """Build enhanced hybrid LSTM-CNN model with additional layers"""
        model = Sequential([
            Input(shape=input_shape),
            
            # First LSTM layer with BatchNormalization
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # CNN layers
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            
            Flatten(),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(len(self.columns))
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        return model
    
    def save_model(self, output_dir):
        """Save the trained model and scaler to the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model (add .keras extension)
        model_path = os.path.join(output_dir, 'model.keras')
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model and scaler saved to {output_dir}")  # Use print as fallback
        return True
    
    def load_model(self, model_dir):
        """Load a previously trained model and scaler"""
        try:
            # Load Keras model
            model_path = os.path.join(model_dir, "model")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            self.scaler = joblib.load(scaler_path)
            
            print(f"Model and scaler loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def fit(self, df):
        """Train the model with enhanced features"""
        print("Preparing data...")
        # Select and prepare features
        data = df[self.columns].copy()
        
        # Handle missing values more robustly
        for col in data.columns:
            if data[col].isnull().any():
                # Forward fill first
                data[col] = data[col].ffill()
                # Then backward fill any remaining NaNs
                data[col] = data[col].bfill()
                # If still any NaNs (beginning of series), fill with 0
                data[col] = data[col].fillna(0)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training model...")
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Build and train model
        self.model = self.build_model((self.sequence_length, len(self.columns)))
        history = self.model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate metrics
        predictions = self.model.predict(X_test, verbose=0)
        eth_price_idx = 0  # Index of Close price in features
        
        mse = mean_squared_error(y_test[:, eth_price_idx], predictions[:, eth_price_idx])
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test[:, eth_price_idx], predictions[:, eth_price_idx])
        mape = self.mean_absolute_percentage_error(y_test[:, eth_price_idx], predictions[:, eth_price_idx])
        da = self.directional_accuracy(y_test[:, eth_price_idx], predictions[:, eth_price_idx])
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'da': da
        }
        
        print(f"\nMetrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Directional Accuracy: {da:.2f}%")
        
        return metrics
    
    def predict_multiple_days(self, df, days_ahead=30):
        """Predict prices for multiple days ahead"""
        try:
            # Prepare the initial sequence
            data = df[self.columns].copy()
            
            # Handle missing values
            for col in data.columns:
                if data[col].isnull().any():
                    data[col] = data[col].ffill()
                    data[col] = data[col].bfill()
                    data[col] = data[col].fillna(0)
            
            scaled_data = self.scaler.transform(data)
            current_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            for i in range(days_ahead):
                # Reshape sequence for prediction
                sequence = current_sequence.reshape(1, self.sequence_length, len(self.columns))
                
                # Make prediction
                pred = self.model.predict(sequence, verbose=0)
                
                # Store prediction
                predictions.append(pred[0])
                
                # Update sequence for next prediction
                current_sequence = np.vstack((current_sequence[1:], pred[0]))
            
            # Convert predictions back to original scale
            predictions_array = np.array(predictions)
            predictions_full = self.scaler.inverse_transform(predictions_array)
            
            # Return just the predicted prices (first column)
            return predictions_full[:, 0]
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
            
    def feature_importance(self, df, n_samples=1000, top_n=10):
        """Calculate feature importance by permutation
        Note: This doesn't affect model training - it's an analysis tool
        """
        if self.model is None:
            print("No model found. Train or load a model first.")
            return None
            
        print("Calculating feature importance (this may take a while)...")
        
        # Prepare data
        data = df[self.columns].copy()
        for col in data.columns:
            if data[col].isnull().any():
                data[col] = data[col].ffill()
                data[col] = data[col].bfill()
                data[col] = data[col].fillna(0)
                
        scaled_data = self.scaler.transform(data)
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        
        # Use a subset for speed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
            
        # Get baseline error
        baseline_preds = self.model.predict(X_sample, verbose=0)
        baseline_error = mean_squared_error(y_sample[:, 0], baseline_preds[:, 0])
        
        # Calculate importance for each feature
        importance = {}
        for i, feature_name in enumerate(self.columns):
            # Make a copy of the data
            X_permuted = X_sample.copy()
            
            # Shuffle the feature across all sequences
            for seq_idx in range(len(X_permuted)):
                np.random.shuffle(X_permuted[seq_idx, :, i])
                
            # Predict with permuted feature
            perm_preds = self.model.predict(X_permuted, verbose=0)
            perm_error = mean_squared_error(y_sample[:, 0], perm_preds[:, 0])
            
            # Importance = increase in error when feature is permuted
            importance[feature_name] = perm_error - baseline_error
            
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(
            importance.items(), key=lambda item: item[1], reverse=True)}
        
        # Return top N features
        top_features = {k: v for i, (k, v) in enumerate(sorted_importance.items()) if i < top_n}
        
        return top_features