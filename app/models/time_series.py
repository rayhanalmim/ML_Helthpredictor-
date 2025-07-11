import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class VitalSignsPredictor:
    def __init__(self):
        self.lstm_models = {}
        self.arima_models = {}
        self.scalers = {}
        self.sequence_length = 10  # Look back 10 time steps
        self.is_trained = False
        
        # Generate sample time series data for training
        self._generate_sample_data()
        
    def _generate_sample_data(self):
        """Generate synthetic time series data for vital signs"""
        np.random.seed(42)
        
        # Generate 30 days of hourly data
        timestamps = pd.date_range(start='2024-01-01', periods=720, freq='H')
        
        # Heart rate with circadian rhythm and noise
        base_hr = 72
        circadian_hr = 8 * np.sin(2 * np.pi * np.arange(720) / 24)  # Daily cycle
        noise_hr = np.random.normal(0, 3, 720)
        trend_hr = np.linspace(0, 2, 720)  # Slight upward trend
        heart_rate = base_hr + circadian_hr + noise_hr + trend_hr
        heart_rate = np.clip(heart_rate, 55, 95)
        
        # Blood pressure with similar patterns
        base_sys = 120
        circadian_sys = 6 * np.sin(2 * np.pi * np.arange(720) / 24 + np.pi/4)
        noise_sys = np.random.normal(0, 4, 720)
        systolic_bp = base_sys + circadian_sys + noise_sys
        systolic_bp = np.clip(systolic_bp, 100, 150)
        
        base_dia = 80
        diastolic_bp = systolic_bp * 0.67 + np.random.normal(0, 2, 720)
        diastolic_bp = np.clip(diastolic_bp, 60, 100)
        
        # Glucose with meal-related spikes
        base_glucose = 100
        meal_spikes = np.zeros(720)
        # Add spikes at typical meal times (8am, 1pm, 7pm)
        for i in range(720):
            hour = i % 24
            if hour in [8, 13, 19]:  # Meal times
                meal_spikes[i:i+3] = [20, 15, 5] if i+3 <= 720 else [20, 15, 5][:720-i]
        
        noise_glucose = np.random.normal(0, 5, 720)
        glucose = base_glucose + meal_spikes + noise_glucose
        glucose = np.clip(glucose, 80, 180)
        
        # Temperature with daily variations
        base_temp = 98.6
        circadian_temp = 1.2 * np.sin(2 * np.pi * np.arange(720) / 24 + np.pi)
        noise_temp = np.random.normal(0, 0.3, 720)
        temperature = base_temp + circadian_temp + noise_temp
        temperature = np.clip(temperature, 97.0, 100.5)
        
        self.sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': heart_rate,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'glucose': glucose,
            'temperature': temperature
        })
    
    def _prepare_lstm_data(self, data, target_column):
        """Prepare data for LSTM training"""
        values = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _fit_arima_model(self, data, target_column):
        """Fit ARIMA model to time series data"""
        ts_data = data[target_column].values
        
        # Check stationarity
        adf_result = adfuller(ts_data)
        is_stationary = adf_result[1] < 0.05
        
        # If not stationary, difference the data
        if not is_stationary:
            ts_data = np.diff(ts_data)
        
        # Find optimal ARIMA parameters (simplified approach)
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        # Fit final model
        final_model = ARIMA(ts_data, order=best_order)
        fitted_final_model = final_model.fit()
        
        return fitted_final_model, is_stationary
    
    def train_models(self, data=None):
        """Train both LSTM and ARIMA models"""
        if data is None:
            data = self.sample_data
        
        target_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'temperature']
        
        for column in target_columns:
            print(f"Training models for {column}...")
            
            # Train LSTM model
            try:
                X, y, scaler = self._prepare_lstm_data(data, column)
                
                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Build and train LSTM
                lstm_model = self._build_lstm_model((X.shape[1], 1))
                lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                             validation_data=(X_test, y_test), verbose=0)
                
                self.lstm_models[column] = lstm_model
                self.scalers[column] = scaler
                
                # Calculate LSTM performance
                y_pred = lstm_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print(f"LSTM MSE for {column}: {mse:.4f}")
                
            except Exception as e:
                print(f"Error training LSTM for {column}: {e}")
            
            # Train ARIMA model
            try:
                arima_model, is_stationary = self._fit_arima_model(data, column)
                self.arima_models[column] = {
                    'model': arima_model,
                    'is_stationary': is_stationary
                }
                print(f"ARIMA model trained for {column}")
                
            except Exception as e:
                print(f"Error training ARIMA for {column}: {e}")
        
        self.is_trained = True
        print("All models trained successfully!")
    
    def predict_heart_rate(self, data, steps=7):
        """Predict heart rate for next 'steps' time periods"""
        if not self.is_trained:
            self.train_models()
        
        target_column = 'heart_rate'
        
        # LSTM prediction
        lstm_predictions = self._predict_with_lstm(data, target_column, steps)
        
        # ARIMA prediction
        arima_predictions = self._predict_with_arima(data, target_column, steps)
        
        # Ensemble: average of both predictions
        if lstm_predictions is not None and arima_predictions is not None:
            final_predictions = (lstm_predictions + arima_predictions) / 2
        elif lstm_predictions is not None:
            final_predictions = lstm_predictions
        elif arima_predictions is not None:
            final_predictions = arima_predictions
        else:
            # Fallback: simple trend extrapolation
            recent_values = data[target_column].tail(steps).values
            trend = np.mean(np.diff(recent_values))
            final_predictions = [recent_values[-1] + trend * i for i in range(1, steps + 1)]
        
        return np.array(final_predictions).flatten()
    
    def predict_blood_pressure(self, data, steps=7):
        """Predict blood pressure for next 'steps' time periods"""
        if not self.is_trained:
            self.train_models()
        
        systolic_pred = self._predict_with_ensemble(data, 'systolic_bp', steps)
        diastolic_pred = self._predict_with_ensemble(data, 'diastolic_bp', steps)
        
        return {
            'systolic': systolic_pred,
            'diastolic': diastolic_pred
        }
    
    def predict_glucose(self, data, steps=7):
        """Predict glucose levels for next 'steps' time periods"""
        if not self.is_trained:
            self.train_models()
        
        return self._predict_with_ensemble(data, 'glucose', steps)
    
    def predict_temperature(self, data, steps=7):
        """Predict temperature for next 'steps' time periods"""
        if not self.is_trained:
            self.train_models()
        
        return self._predict_with_ensemble(data, 'temperature', steps)
    
    def _predict_with_lstm(self, data, target_column, steps):
        """Make predictions using LSTM model"""
        if target_column not in self.lstm_models:
            return None
        
        try:
            model = self.lstm_models[target_column]
            scaler = self.scalers[target_column]
            
            # Prepare last sequence
            values = data[target_column].tail(self.sequence_length).values.reshape(-1, 1)
            scaled_values = scaler.transform(values)
            
            predictions = []
            current_sequence = scaled_values.copy()
            
            for _ in range(steps):
                # Reshape for prediction
                input_seq = current_sequence.reshape((1, self.sequence_length, 1))
                
                # Predict next value
                pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
                
                # Inverse transform
                pred_actual = scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred_actual)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_scaled
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Error in LSTM prediction for {target_column}: {e}")
            return None
    
    def _predict_with_arima(self, data, target_column, steps):
        """Make predictions using ARIMA model"""
        if target_column not in self.arima_models:
            return None
        
        try:
            arima_info = self.arima_models[target_column]
            model = arima_info['model']
            
            # Get forecast
            forecast = model.forecast(steps=steps)
            
            return np.array(forecast)
            
        except Exception as e:
            print(f"Error in ARIMA prediction for {target_column}: {e}")
            return None
    
    def _predict_with_ensemble(self, data, target_column, steps):
        """Make predictions using ensemble of LSTM and ARIMA"""
        lstm_pred = self._predict_with_lstm(data, target_column, steps)
        arima_pred = self._predict_with_arima(data, target_column, steps)
        
        if lstm_pred is not None and arima_pred is not None:
            return (lstm_pred + arima_pred) / 2
        elif lstm_pred is not None:
            return lstm_pred
        elif arima_pred is not None:
            return arima_pred
        else:
            # Fallback prediction
            recent_values = data[target_column].tail(steps).values
            if len(recent_values) > 1:
                trend = np.mean(np.diff(recent_values))
                return np.array([recent_values[-1] + trend * i for i in range(1, steps + 1)])
            else:
                return np.array([recent_values[-1]] * steps)
    
    def detect_anomalies_in_predictions(self, data, target_column, threshold=2.0):
        """Detect anomalies in vital sign predictions"""
        predictions = self._predict_with_ensemble(data, target_column, 24)  # 24 hours ahead
        
        if predictions is None:
            return []
        
        # Calculate normal range based on historical data
        historical_values = data[target_column].values
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        anomalies = []
        for i, pred in enumerate(predictions):
            z_score = abs(pred - mean_val) / std_val
            if z_score > threshold:
                anomalies.append({
                    'time_ahead': i + 1,
                    'predicted_value': pred,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium'
                })
        
        return anomalies
    
    def get_prediction_confidence(self, data, target_column, steps=7):
        """Calculate confidence intervals for predictions"""
        # Generate multiple predictions with different random seeds
        predictions_list = []
        
        for seed in range(10):
            np.random.seed(seed)
            pred = self._predict_with_ensemble(data, target_column, steps)
            if pred is not None:
                predictions_list.append(pred)
        
        if not predictions_list:
            return None
        
        predictions_array = np.array(predictions_list)
        
        # Calculate confidence intervals
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        confidence_intervals = {
            'mean': mean_pred,
            'lower_95': mean_pred - 1.96 * std_pred,
            'upper_95': mean_pred + 1.96 * std_pred,
            'lower_68': mean_pred - std_pred,
            'upper_68': mean_pred + std_pred
        }
        
        return confidence_intervals
    
    def analyze_trends(self, data, target_column):
        """Analyze trends in vital signs data"""
        values = data[target_column].values
        timestamps = data['timestamp'] if 'timestamp' in data.columns else range(len(values))
        
        # Calculate basic trend metrics
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Moving average trend
        window_size = min(7, len(values) // 3)
        if window_size >= 2:
            moving_avg = pd.Series(values).rolling(window=window_size).mean()
            recent_avg = moving_avg.tail(window_size//2).mean()
            earlier_avg = moving_avg.head(len(moving_avg)//2).mean()
            ma_trend = recent_avg - earlier_avg
        else:
            ma_trend = slope
        
        # Volatility
        volatility = np.std(values)
        
        # Trend classification
        if slope > 0.1:
            trend_direction = 'increasing'
        elif slope < -0.1:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'moving_average_trend': ma_trend,
            'volatility': volatility,
            'current_value': values[-1],
            'average_value': np.mean(values),
            'min_value': np.min(values),
            'max_value': np.max(values)
        }
    
    def get_health_insights(self, data):
        """Generate health insights based on time series analysis"""
        insights = []
        
        vital_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'temperature']
        
        for column in vital_columns:
            if column not in data.columns:
                continue
                
            trend_analysis = self.analyze_trends(data, column)
            
            # Generate insights based on trends
            if column == 'heart_rate':
                if trend_analysis['trend'] == 'increasing' and trend_analysis['slope'] > 0.5:
                    insights.append(f"Heart rate showing upward trend - consider stress management")
                elif trend_analysis['volatility'] > 10:
                    insights.append(f"High heart rate variability detected - monitor closely")
                    
            elif column == 'systolic_bp':
                if trend_analysis['average_value'] > 140:
                    insights.append(f"Average blood pressure elevated - consult healthcare provider")
                elif trend_analysis['trend'] == 'increasing':
                    insights.append(f"Blood pressure trending upward - monitor diet and exercise")
                    
            elif column == 'glucose':
                if trend_analysis['average_value'] > 126:
                    insights.append(f"Average glucose levels elevated - consider dietary changes")
                elif trend_analysis['volatility'] > 20:
                    insights.append(f"High glucose variability - review meal timing and composition")
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = VitalSignsPredictor()
    
    # Train models
    predictor.train_models()
    
    # Example predictions
    sample_data = predictor.sample_data.tail(50)  # Use last 50 data points
    
    # Predict heart rate
    hr_predictions = predictor.predict_heart_rate(sample_data, steps=24)
    print(f"Heart rate predictions for next 24 hours: {hr_predictions[:5]}...")
    
    # Predict blood pressure
    bp_predictions = predictor.predict_blood_pressure(sample_data, steps=12)
    print(f"Systolic BP predictions: {bp_predictions['systolic'][:3]}...")
    
    # Get health insights
    insights = predictor.get_health_insights(sample_data)
    print(f"Health insights: {insights}")
    
    # Check for anomalies in predictions
    anomalies = predictor.detect_anomalies_in_predictions(sample_data, 'heart_rate')
    print(f"Detected {len(anomalies)} potential anomalies in heart rate predictions")