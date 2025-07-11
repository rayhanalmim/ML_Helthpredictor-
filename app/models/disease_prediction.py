import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize different models for ensemble
        self.base_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
        # Disease-specific models
        self.disease_models = {
            'heart_disease': None,
            'diabetes': None,
            'hypertension': None,
            'stroke': None,
            'kidney_disease': None
        }
        
        # Generate synthetic training data
        self._generate_training_data()
        self._train_models()
    
    def _generate_training_data(self):
        """Generate synthetic medical data for training"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate features
        ages = np.random.normal(50, 15, n_samples)
        ages = np.clip(ages, 18, 90)
        
        genders = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
        
        # BMI
        bmis = np.random.normal(25, 5, n_samples)
        bmis = np.clip(bmis, 15, 50)
        
        # Blood pressure
        systolic_bp = np.random.normal(120, 20, n_samples)
        systolic_bp = np.clip(systolic_bp, 80, 200)
        
        diastolic_bp = np.random.normal(80, 15, n_samples)
        diastolic_bp = np.clip(diastolic_bp, 50, 120)
        
        # Cholesterol
        cholesterol = np.random.normal(200, 40, n_samples)
        cholesterol = np.clip(cholesterol, 100, 400)
        
        # Heart rate
        heart_rates = np.random.normal(75, 15, n_samples)
        heart_rates = np.clip(heart_rates, 50, 150)
        
        # Glucose
        glucose = np.random.normal(100, 25, n_samples)
        glucose = np.clip(glucose, 70, 300)
        
        # Exercise habits (0-5 scale)
        exercise = np.random.poisson(2, n_samples)
        exercise = np.clip(exercise, 0, 5)
        
        # Smoking (0: No, 1: Yes)
        smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Family history (0: No, 1: Yes)
        family_history = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'age': ages,
            'gender': genders,
            'bmi': bmis,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'cholesterol': cholesterol,
            'heart_rate': heart_rates,
            'glucose': glucose,
            'exercise': exercise,
            'smoking': smoking,
            'family_history': family_history
        })
        
        # Generate target variables (diseases)
        self._generate_disease_targets()
    
    def _generate_disease_targets(self):
        """Generate disease targets based on realistic risk factors"""
        n_samples = len(self.training_data)
        
        # Heart Disease
        heart_disease_prob = (
            0.1 + 
            0.3 * (self.training_data['age'] > 60) +
            0.2 * (self.training_data['gender'] == 1) +
            0.2 * (self.training_data['cholesterol'] > 240) +
            0.3 * (self.training_data['systolic_bp'] > 140) +
            0.3 * self.training_data['smoking'] +
            0.2 * self.training_data['family_history'] +
            0.1 * (self.training_data['bmi'] > 30)
        )
        heart_disease_prob = np.clip(heart_disease_prob, 0, 1)
        self.training_data['heart_disease'] = np.random.binomial(1, heart_disease_prob)
        
        # Diabetes
        diabetes_prob = (
            0.05 +
            0.2 * (self.training_data['age'] > 45) +
            0.3 * (self.training_data['bmi'] > 30) +
            0.4 * (self.training_data['glucose'] > 126) +
            0.1 * self.training_data['family_history'] +
            0.1 * (self.training_data['exercise'] < 2)
        )
        diabetes_prob = np.clip(diabetes_prob, 0, 1)
        self.training_data['diabetes'] = np.random.binomial(1, diabetes_prob)
        
        # Hypertension
        hypertension_prob = (
            0.1 +
            0.3 * (self.training_data['systolic_bp'] > 140) +
            0.2 * (self.training_data['age'] > 50) +
            0.2 * (self.training_data['bmi'] > 25) +
            0.1 * self.training_data['smoking'] +
            0.1 * self.training_data['family_history']
        )
        hypertension_prob = np.clip(hypertension_prob, 0, 1)
        self.training_data['hypertension'] = np.random.binomial(1, hypertension_prob)
        
        # Stroke
        stroke_prob = (
            0.02 +
            0.3 * (self.training_data['age'] > 65) +
            0.2 * (self.training_data['systolic_bp'] > 160) +
            0.3 * self.training_data['smoking'] +
            0.2 * self.training_data['heart_disease'] +
            0.1 * self.training_data['diabetes']
        )
        stroke_prob = np.clip(stroke_prob, 0, 1)
        self.training_data['stroke'] = np.random.binomial(1, stroke_prob)
        
        # Kidney Disease
        kidney_disease_prob = (
            0.05 +
            0.2 * (self.training_data['age'] > 60) +
            0.3 * self.training_data['diabetes'] +
            0.2 * self.training_data['hypertension'] +
            0.1 * self.training_data['family_history']
        )
        kidney_disease_prob = np.clip(kidney_disease_prob, 0, 1)
        self.training_data['kidney_disease'] = np.random.binomial(1, kidney_disease_prob)
    
    def _train_models(self):
        """Train models for each disease"""
        feature_columns = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                          'cholesterol', 'heart_rate', 'glucose', 'exercise', 
                          'smoking', 'family_history']
        
        X = self.training_data[feature_columns]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        diseases = ['heart_disease', 'diabetes', 'hypertension', 'stroke', 'kidney_disease']
        
        for disease in diseases:
            y = self.training_data[disease]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models and select the best one
            best_model = None
            best_score = 0
            
            for model_name, model in self.base_models.items():
                try:
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    print(f"Error training {model_name} for {disease}: {e}")
                    continue
            
            self.disease_models[disease] = best_model
            print(f"Best model for {disease}: {type(best_model).__name__} (Accuracy: {best_score:.3f})")
        
        self.is_trained = True
    
    def predict_single_disease(self, user_data, disease):
        """Predict probability of a single disease"""
        if not self.is_trained or disease not in self.disease_models:
            return 0.5
        
        # Prepare input data
        feature_columns = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                          'cholesterol', 'heart_rate', 'glucose', 'exercise', 
                          'smoking', 'family_history']
        
        # Fill missing values with defaults
        processed_data = {}
        for col in feature_columns:
            if col in user_data:
                processed_data[col] = user_data[col]
            else:
                # Default values
                defaults = {
                    'age': 30, 'gender': 0, 'bmi': 25, 'systolic_bp': 120,
                    'diastolic_bp': 80, 'cholesterol': 200, 'heart_rate': 75,
                    'glucose': 100, 'exercise': 2, 'smoking': 0, 'family_history': 0
                }
                processed_data[col] = defaults[col]
        
        # Convert to array and scale
        input_array = np.array([[processed_data[col] for col in feature_columns]])
        input_scaled = self.scaler.transform(input_array)
        
        # Get prediction
        model = self.disease_models[disease]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = model.predict(input_scaled)[0]
        
        return probability
    
    def predict_multiple_diseases(self, user_data):
        """Predict probabilities for all diseases"""
        diseases = ['heart_disease', 'diabetes', 'hypertension', 'stroke', 'kidney_disease']
        
        predictions = {}
        for disease in diseases:
            predictions[disease.replace('_', ' ').title()] = self.predict_single_disease(user_data, disease)
        
        return predictions
    
    def get_risk_factors(self, user_data, disease):
        """Get risk factors for a specific disease"""
        risk_factors = []
        
        if disease == 'heart_disease':
            if user_data.get('age', 0) > 60:
                risk_factors.append("Age over 60")
            if user_data.get('cholesterol', 0) > 240:
                risk_factors.append("High cholesterol")
            if user_data.get('systolic_bp', 0) > 140:
                risk_factors.append("High blood pressure")
            if user_data.get('smoking', 0) == 1:
                risk_factors.append("Smoking")
            if user_data.get('bmi', 0) > 30:
                risk_factors.append("Obesity")
                
        elif disease == 'diabetes':
            if user_data.get('bmi', 0) > 30:
                risk_factors.append("Obesity")
            if user_data.get('glucose', 0) > 126:
                risk_factors.append("High blood glucose")
            if user_data.get('age', 0) > 45:
                risk_factors.append("Age over 45")
            if user_data.get('exercise', 0) < 2:
                risk_factors.append("Low physical activity")
                
        elif disease == 'hypertension':
            if user_data.get('systolic_bp', 0) > 140:
                risk_factors.append("High systolic blood pressure")
            if user_data.get('bmi', 0) > 25:
                risk_factors.append("Overweight")
            if user_data.get('age', 0) > 50:
                risk_factors.append("Age over 50")
            if user_data.get('smoking', 0) == 1:
                risk_factors.append("Smoking")
        
        return risk_factors
    
    def get_prevention_tips(self, disease):
        """Get prevention tips for a specific disease"""
        tips = {
            'heart_disease': [
                "Maintain a healthy diet low in saturated fats",
                "Exercise regularly (at least 150 minutes per week)",
                "Don't smoke and limit alcohol consumption",
                "Manage stress through relaxation techniques",
                "Monitor blood pressure and cholesterol regularly"
            ],
            'diabetes': [
                "Maintain a healthy weight",
                "Eat a balanced diet with controlled carbohydrates",
                "Exercise regularly to improve insulin sensitivity",
                "Monitor blood glucose levels",
                "Limit processed foods and sugary drinks"
            ],
            'hypertension': [
                "Reduce sodium intake",
                "Maintain a healthy weight",
                "Exercise regularly",
                "Limit alcohol consumption",
                "Manage stress effectively"
            ],
            'stroke': [
                "Control blood pressure",
                "Don't smoke",
                "Manage diabetes if present",
                "Exercise regularly",
                "Eat a diet rich in fruits and vegetables"
            ],
            'kidney_disease': [
                "Control blood pressure and diabetes",
                "Drink plenty of water",
                "Limit protein intake if advised by doctor",
                "Avoid NSAIDs when possible",
                "Regular kidney function monitoring"
            ]
        }
        
        return tips.get(disease, ["Maintain a healthy lifestyle", "Regular medical checkups"])
    
    def save_models(self, filepath):
        """Save trained models to file"""
        model_data = {
            'disease_models': self.disease_models,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath):
        """Load trained models from file"""
        try:
            model_data = joblib.load(filepath)
            self.disease_models = model_data['disease_models']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_performance(self):
        """Get performance metrics for all models"""
        if not self.is_trained:
            return None
        
        feature_columns = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 
                          'cholesterol', 'heart_rate', 'glucose', 'exercise', 
                          'smoking', 'family_history']
        
        X = self.training_data[feature_columns]
        X_scaled = self.scaler.transform(X)
        
        performance = {}
        diseases = ['heart_disease', 'diabetes', 'hypertension', 'stroke', 'kidney_disease']
        
        for disease in diseases:
            y = self.training_data[disease]
            model = self.disease_models[disease]
            
            if model:
                predictions = model.predict(X_scaled)
                accuracy = accuracy_score(y, predictions)
                performance[disease] = {
                    'accuracy': accuracy,
                    'model_type': type(model).__name__
                }
        
        return performance

# Example usage
if __name__ == "__main__":
    # Initialize the model
    disease_predictor = DiseasePredictionModel()
    
    # Example prediction
    user_data = {
        'age': 55,
        'gender': 1,  # Male
        'bmi': 28,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'cholesterol': 220,
        'heart_rate': 80,
        'glucose': 110,
        'exercise': 2,
        'smoking': 0,
        'family_history': 1
    }
    
    predictions = disease_predictor.predict_multiple_diseases(user_data)
    print("Disease Risk Predictions:")
    for disease, risk in predictions.items():
        print(f"{disease}: {risk:.1%} risk")