import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.1)
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Health parameter normal ranges
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'glucose': (70, 140),
            'temperature': (97.0, 99.5),
            'cholesterol': (125, 200),
            'bmi': (18.5, 25.0)
        }
        
        # Generate training data
        self._generate_training_data()
        self._train_models()
    
    def _generate_training_data(self):
        """Generate synthetic health data for training anomaly detection models"""
        np.random.seed(42)
        n_normal = 5000
        n_anomalies = 200
        
        # Generate normal health data
        normal_data = []
        for _ in range(n_normal):
            data_point = {
                'heart_rate': np.random.normal(75, 10),
                'systolic_bp': np.random.normal(120, 15),
                'diastolic_bp': np.random.normal(80, 10),
                'glucose': np.random.normal(100, 15),
                'temperature': np.random.normal(98.6, 0.8),
                'cholesterol': np.random.normal(180, 30),
                'bmi': np.random.normal(23, 3)
            }
            
            # Add some correlation between features
            data_point['systolic_bp'] += data_point['bmi'] * 2
            data_point['diastolic_bp'] = data_point['systolic_bp'] * 0.65 + np.random.normal(0, 5)
            data_point['heart_rate'] += (data_point['bmi'] - 23) * 1.5
            
            normal_data.append(data_point)
        
        # Generate anomalous data
        anomalous_data = []
        for _ in range(n_anomalies):
            # Create anomalies by sampling from extreme distributions
            anomaly_type = np.random.choice(['high_bp', 'low_hr', 'high_glucose', 'fever', 'multiple'])
            
            if anomaly_type == 'high_bp':
                data_point = {
                    'heart_rate': np.random.normal(75, 10),
                    'systolic_bp': np.random.normal(180, 20),  # High BP
                    'diastolic_bp': np.random.normal(110, 15),  # High BP
                    'glucose': np.random.normal(100, 15),
                    'temperature': np.random.normal(98.6, 0.8),
                    'cholesterol': np.random.normal(180, 30),
                    'bmi': np.random.normal(23, 3)
                }
            elif anomaly_type == 'low_hr':
                data_point = {
                    'heart_rate': np.random.normal(45, 5),  # Low heart rate
                    'systolic_bp': np.random.normal(120, 15),
                    'diastolic_bp': np.random.normal(80, 10),
                    'glucose': np.random.normal(100, 15),
                    'temperature': np.random.normal(98.6, 0.8),
                    'cholesterol': np.random.normal(180, 30),
                    'bmi': np.random.normal(23, 3)
                }
            elif anomaly_type == 'high_glucose':
                data_point = {
                    'heart_rate': np.random.normal(75, 10),
                    'systolic_bp': np.random.normal(120, 15),
                    'diastolic_bp': np.random.normal(80, 10),
                    'glucose': np.random.normal(250, 30),  # High glucose
                    'temperature': np.random.normal(98.6, 0.8),
                    'cholesterol': np.random.normal(180, 30),
                    'bmi': np.random.normal(23, 3)
                }
            elif anomaly_type == 'fever':
                data_point = {
                    'heart_rate': np.random.normal(90, 10),  # Elevated due to fever
                    'systolic_bp': np.random.normal(120, 15),
                    'diastolic_bp': np.random.normal(80, 10),
                    'glucose': np.random.normal(100, 15),
                    'temperature': np.random.normal(102, 1),  # Fever
                    'cholesterol': np.random.normal(180, 30),
                    'bmi': np.random.normal(23, 3)
                }
            else:  # multiple anomalies
                data_point = {
                    'heart_rate': np.random.normal(110, 15),  # High
                    'systolic_bp': np.random.normal(160, 20),  # High
                    'diastolic_bp': np.random.normal(100, 15),  # High
                    'glucose': np.random.normal(180, 25),  # High
                    'temperature': np.random.normal(98.6, 0.8),
                    'cholesterol': np.random.normal(280, 40),  # High
                    'bmi': np.random.normal(32, 4)  # High
                }
            
            anomalous_data.append(data_point)
        
        # Combine data
        all_data = normal_data + anomalous_data
        self.training_data = pd.DataFrame(all_data)
        
        # Ensure realistic ranges
        self.training_data = self.training_data.clip(lower={
            'heart_rate': 30, 'systolic_bp': 70, 'diastolic_bp': 40,
            'glucose': 50, 'temperature': 95, 'cholesterol': 100, 'bmi': 15
        })
        
        self.training_data = self.training_data.clip(upper={
            'heart_rate': 200, 'systolic_bp': 250, 'diastolic_bp': 150,
            'glucose': 400, 'temperature': 106, 'cholesterol': 400, 'bmi': 50
        })
        
        # Create labels (0 for normal, 1 for anomaly)
        self.labels = np.array([0] * n_normal + [1] * n_anomalies)
    
    def _build_autoencoder(self, input_dim):
        """Build autoencoder neural network"""
        encoding_dim = max(2, input_dim // 2)
        
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        encoder = Dense(encoding_dim // 2, activation="relu")(encoder)
        
        decoder = Dense(encoding_dim, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="linear")(decoder)
        
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def _train_models(self):
        """Train all anomaly detection models"""
        # Prepare data
        feature_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 
                          'glucose', 'temperature', 'cholesterol', 'bmi']
        X = self.training_data[feature_columns].values
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train on normal data only for unsupervised methods
        normal_indices = self.labels == 0
        X_normal = X_scaled[normal_indices]
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_normal)
        
        # Train One-Class SVM
        self.one_class_svm.fit(X_normal)
        
        # Train Autoencoder
        self.autoencoder = self._build_autoencoder(X_scaled.shape[1])
        self.autoencoder.fit(X_normal, X_normal, 
                            epochs=100, batch_size=32, 
                            validation_split=0.1, verbose=0)
        
        self.is_trained = True
        
        # Evaluate models
        self._evaluate_models(X_scaled)
    
    def _evaluate_models(self, X_scaled):
        """Evaluate anomaly detection models"""
        # Isolation Forest predictions
        if_predictions = self.isolation_forest.predict(X_scaled)
        if_predictions = (if_predictions == -1).astype(int)  # Convert to 0/1
        
        # One-Class SVM predictions
        svm_predictions = self.one_class_svm.predict(X_scaled)
        svm_predictions = (svm_predictions == -1).astype(int)
        
        # Autoencoder predictions
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        ae_threshold = np.percentile(mse, 90)  # Top 10% as anomalies
        ae_predictions = (mse > ae_threshold).astype(int)
        
        # Calculate accuracy for each model
        if_accuracy = np.mean(if_predictions == self.labels)
        svm_accuracy = np.mean(svm_predictions == self.labels)
        ae_accuracy = np.mean(ae_predictions == self.labels)
        
        print(f"Isolation Forest Accuracy: {if_accuracy:.3f}")
        print(f"One-Class SVM Accuracy: {svm_accuracy:.3f}")
        print(f"Autoencoder Accuracy: {ae_accuracy:.3f}")
        
        # Store threshold for autoencoder
        self.ae_threshold = ae_threshold
    
    def detect_anomalies(self, data):
        """Detect anomalies in health data"""
        if not self.is_trained:
            return {'has_anomalies': False, 'anomaly_details': []}
        
        # Prepare data
        feature_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 
                          'glucose', 'temperature']
        
        # Fill missing columns with defaults
        for col in feature_columns:
            if col not in data.columns:
                defaults = {
                    'heart_rate': 75, 'systolic_bp': 120, 'diastolic_bp': 80,
                    'glucose': 100, 'temperature': 98.6
                }
                data[col] = defaults[col]
        
        # Add derived features if missing
        if 'cholesterol' not in data.columns:
            data['cholesterol'] = 180  # Default value
        if 'bmi' not in data.columns:
            data['bmi'] = 23  # Default value
        
        recent_data = data[['heart_rate', 'systolic_bp', 'diastolic_bp', 
                           'glucose', 'temperature', 'cholesterol', 'bmi']].tail(10)
        
        anomalies_detected = []
        
        for idx, row in recent_data.iterrows():
            row_data = row.values.reshape(1, -1)
            row_scaled = self.scaler.transform(row_data)
            
            # Check each detection method
            anomaly_scores = {}
            
            # Isolation Forest
            if_score = self.isolation_forest.decision_function(row_scaled)[0]
            if_anomaly = self.isolation_forest.predict(row_scaled)[0] == -1
            anomaly_scores['isolation_forest'] = {'score': if_score, 'is_anomaly': if_anomaly}
            
            # One-Class SVM
            svm_score = self.one_class_svm.decision_function(row_scaled)[0]
            svm_anomaly = self.one_class_svm.predict(row_scaled)[0] == -1
            anomaly_scores['one_class_svm'] = {'score': svm_score, 'is_anomaly': svm_anomaly}
            
            # Autoencoder
            reconstruction = self.autoencoder.predict(row_scaled, verbose=0)
            mse = np.mean(np.square(row_scaled - reconstruction))
            ae_anomaly = mse > self.ae_threshold
            anomaly_scores['autoencoder'] = {'score': mse, 'is_anomaly': ae_anomaly}
            
            # Statistical anomaly detection
            stat_anomalies = self._detect_statistical_anomalies(row)
            
            # Ensemble decision (majority voting)
            anomaly_votes = sum([
                anomaly_scores['isolation_forest']['is_anomaly'],
                anomaly_scores['one_class_svm']['is_anomaly'],
                anomaly_scores['autoencoder']['is_anomaly']
            ])
            
            is_anomaly = anomaly_votes >= 2 or len(stat_anomalies) > 0
            
            if is_anomaly:
                anomaly_details = {
                    'timestamp': idx if hasattr(idx, 'timestamp') else len(anomalies_detected),
                    'overall_anomaly': True,
                    'methods_detected': [],
                    'severity': self._calculate_severity(row, anomaly_scores),
                    'affected_vitals': [],
                    'recommendations': []
                }
                
                # Add detection methods
                if anomaly_scores['isolation_forest']['is_anomaly']:
                    anomaly_details['methods_detected'].append('Isolation Forest')
                if anomaly_scores['one_class_svm']['is_anomaly']:
                    anomaly_details['methods_detected'].append('One-Class SVM')
                if anomaly_scores['autoencoder']['is_anomaly']:
                    anomaly_details['methods_detected'].append('Autoencoder')
                if stat_anomalies:
                    anomaly_details['methods_detected'].append('Statistical Analysis')
                    anomaly_details['affected_vitals'].extend([a['vital'] for a in stat_anomalies])
                
                # Add specific vital anomalies
                for vital_anomaly in stat_anomalies:
                    anomaly_details['affected_vitals'].append(vital_anomaly['vital'])
                
                # Generate recommendations
                anomaly_details['recommendations'] = self._generate_anomaly_recommendations(
                    anomaly_details['affected_vitals'], anomaly_details['severity']
                )
                
                anomalies_detected.append(anomaly_details)
        
        return {
            'has_anomalies': len(anomalies_detected) > 0,
            'anomaly_count': len(anomalies_detected),
            'anomaly_details': anomalies_detected
        }
    
    def _detect_statistical_anomalies(self, row):
        """Detect anomalies using statistical thresholds"""
        anomalies = []
        
        for vital, (min_val, max_val) in self.normal_ranges.items():
            if vital in row.index:
                value = row[vital]
                
                if value < min_val:
                    anomalies.append({
                        'vital': vital,
                        'value': value,
                        'type': 'low',
                        'normal_range': f"{min_val}-{max_val}",
                        'severity': self._calculate_vital_severity(vital, value, min_val, max_val)
                    })
                elif value > max_val:
                    anomalies.append({
                        'vital': vital,
                        'value': value,
                        'type': 'high',
                        'normal_range': f"{min_val}-{max_val}",
                        'severity': self._calculate_vital_severity(vital, value, min_val, max_val)
                    })
        
        return anomalies
    
    def _calculate_vital_severity(self, vital, value, min_val, max_val):
        """Calculate severity of vital sign anomaly"""
        if value < min_val:
            deviation = (min_val - value) / min_val
        else:
            deviation = (value - max_val) / max_val
        
        if deviation > 0.5:
            return 'critical'
        elif deviation > 0.2:
            return 'high'
        elif deviation > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_severity(self, row, anomaly_scores):
        """Calculate overall anomaly severity"""
        # Check for critical vital signs
        critical_vitals = []
        
        if row['heart_rate'] < 50 or row['heart_rate'] > 120:
            critical_vitals.append('heart_rate')
        if row['systolic_bp'] > 180 or row['systolic_bp'] < 90:
            critical_vitals.append('blood_pressure')
        if row['glucose'] > 250:
            critical_vitals.append('glucose')
        if row['temperature'] > 102 or row['temperature'] < 96:
            critical_vitals.append('temperature')
        
        if critical_vitals:
            return 'critical'
        
        # Check autoencoder reconstruction error
        ae_score = anomaly_scores['autoencoder']['score']
        if ae_score > 2 * self.ae_threshold:
            return 'high'
        elif ae_score > 1.5 * self.ae_threshold:
            return 'medium'
        else:
            return 'low'
    
    def _generate_anomaly_recommendations(self, affected_vitals, severity):
        """Generate recommendations based on detected anomalies"""
        recommendations = []
        
        if severity == 'critical':
            recommendations.append("Seek immediate medical attention")
            recommendations.append("Contact emergency services if experiencing severe symptoms")
        
        if 'heart_rate' in affected_vitals:
            recommendations.append("Monitor heart rate closely")
            recommendations.append("Avoid strenuous activity until normalized")
            recommendations.append("Consider stress reduction techniques")
        
        if 'blood_pressure' in affected_vitals or 'systolic_bp' in affected_vitals:
            recommendations.append("Monitor blood pressure regularly")
            recommendations.append("Reduce sodium intake")
            recommendations.append("Practice relaxation techniques")
        
        if 'glucose' in affected_vitals:
            recommendations.append("Check blood glucose levels")
            recommendations.append("Review recent food intake")
            recommendations.append("Consider medication timing if diabetic")
        
        if 'temperature' in affected_vitals:
            recommendations.append("Monitor temperature regularly")
            recommendations.append("Stay hydrated")
            recommendations.append("Rest and avoid physical exertion")
        
        if severity in ['high', 'critical']:
            recommendations.append("Contact healthcare provider within 24 hours")
        elif severity == 'medium':
            recommendations.append("Schedule a check-up with healthcare provider")
        
        return recommendations
    
    def predict_future_anomalies(self, data, steps=24):
        """Predict potential future anomalies based on trends"""
        if len(data) < 5:
            return []
        
        feature_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'temperature']
        predictions = []
        
        for col in feature_columns:
            if col in data.columns:
                values = data[col].values
                
                # Simple trend extrapolation
                if len(values) >= 3:
                    # Calculate trend
                    x = np.arange(len(values))
                    trend = np.polyfit(x, values, 1)[0]
                    
                    # Project future values
                    for step in range(1, steps + 1):
                        future_value = values[-1] + trend * step
                        
                        # Check if future value would be anomalous
                        if col in self.normal_ranges:
                            min_val, max_val = self.normal_ranges[col]
                            if future_value < min_val or future_value > max_val:
                                predictions.append({
                                    'vital': col,
                                    'hours_ahead': step,
                                    'predicted_value': future_value,
                                    'type': 'low' if future_value < min_val else 'high',
                                    'confidence': max(0.1, 1.0 - step * 0.1)  # Decreasing confidence
                                })
        
        return predictions
    
    def get_anomaly_summary(self, data):
        """Get summary of anomaly detection results"""
        results = self.detect_anomalies(data)
        
        if not results['has_anomalies']:
            return {
                'status': 'normal',
                'message': 'No anomalies detected in recent health data',
                'risk_level': 'low'
            }
        
        anomaly_count = results['anomaly_count']
        severity_counts = {}
        affected_vitals = set()
        
        for anomaly in results['anomaly_details']:
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            affected_vitals.update(anomaly['affected_vitals'])
        
        # Determine overall risk level
        if 'critical' in severity_counts:
            risk_level = 'critical'
        elif 'high' in severity_counts:
            risk_level = 'high'
        elif 'medium' in severity_counts:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'status': 'anomalies_detected',
            'anomaly_count': anomaly_count,
            'risk_level': risk_level,
            'affected_vitals': list(affected_vitals),
            'severity_breakdown': severity_counts,
            'message': f"Detected {anomaly_count} anomalies with {risk_level} risk level"
        }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = AnomalyDetector()
    
    # Example health data with some anomalies
    test_data = pd.DataFrame({
        'heart_rate': [72, 75, 130, 78, 74],  # 130 is anomalous
        'systolic_bp': [120, 125, 190, 122, 118],  # 190 is anomalous
        'diastolic_bp': [80, 82, 110, 78, 79],  # 110 is anomalous
        'glucose': [100, 105, 102, 280, 98],  # 280 is anomalous
        'temperature': [98.6, 98.4, 98.8, 98.5, 103.2]  # 103.2 is anomalous
    })
    
    # Detect anomalies
    results = detector.detect_anomalies(test_data)
    print(f"Anomalies detected: {results['has_anomalies']}")
    print(f"Number of anomalies: {results['anomaly_count']}")
    
    for anomaly in results['anomaly_details']:
        print(f"Severity: {anomaly['severity']}")
        print(f"Affected vitals: {anomaly['affected_vitals']}")
        print(f"Recommendations: {anomaly['recommendations'][:2]}")
        print("---")