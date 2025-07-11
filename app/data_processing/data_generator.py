import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

class HealthDataGenerator:
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Define realistic health parameter ranges and patterns
        self.health_parameters = {
            'heart_rate': {
                'rest_base': 72,
                'rest_std': 8,
                'exercise_multiplier': 1.5,
                'age_factor': 0.2,
                'stress_factor': 15,
                'range': (50, 120)
            },
            'systolic_bp': {
                'base': 120,
                'std': 12,
                'age_factor': 0.5,
                'stress_factor': 20,
                'exercise_factor': 10,
                'range': (90, 180)
            },
            'diastolic_bp': {
                'base': 80,
                'std': 8,
                'age_factor': 0.3,
                'stress_factor': 10,
                'exercise_factor': 5,
                'range': (60, 110)
            },
            'glucose': {
                'fasting_base': 95,
                'postmeal_base': 140,
                'std': 15,
                'diabetes_factor': 50,
                'stress_factor': 20,
                'range': (70, 200)
            },
            'temperature': {
                'base': 98.6,
                'std': 0.8,
                'fever_threshold': 100.4,
                'circadian_amplitude': 1.0,
                'range': (97.0, 102.0)
            },
            'cholesterol': {
                'base': 190,
                'std': 30,
                'age_factor': 1.0,
                'diet_factor': 40,
                'range': (120, 300)
            },
            'weight': {
                'base': 70,  # kg
                'std': 15,
                'age_factor': 0.5,
                'range': (40, 150)
            }
        }
        
    def generate_sample_vitals(self, num_points=24):
        """Generate sample vital signs data for the last 24 hours"""
        current_time = datetime.now()
        timestamps = [current_time - timedelta(hours=i) for i in range(num_points, 0, -1)]
        
        vitals_data = []
        
        for i, timestamp in enumerate(timestamps):
            # Generate realistic vital signs with some correlation and patterns
            hour = timestamp.hour
            
            # Heart rate with circadian rhythm
            circadian_hr = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak in afternoon
            base_hr = self.health_parameters['heart_rate']['rest_base']
            noise_hr = np.random.normal(0, self.health_parameters['heart_rate']['rest_std'])
            heart_rate = base_hr + circadian_hr + noise_hr
            heart_rate = np.clip(heart_rate, *self.health_parameters['heart_rate']['range'])
            
            # Blood pressure (correlated with heart rate)
            hr_influence = (heart_rate - base_hr) * 0.3
            systolic_bp = (self.health_parameters['systolic_bp']['base'] + 
                          hr_influence + 
                          np.random.normal(0, self.health_parameters['systolic_bp']['std']))
            systolic_bp = np.clip(systolic_bp, *self.health_parameters['systolic_bp']['range'])
            
            diastolic_bp = (self.health_parameters['diastolic_bp']['base'] + 
                           hr_influence * 0.5 + 
                           np.random.normal(0, self.health_parameters['diastolic_bp']['std']))
            diastolic_bp = np.clip(diastolic_bp, *self.health_parameters['diastolic_bp']['range'])
            
            # Glucose with meal spikes
            meal_spike = 0
            if hour in [8, 13, 19]:  # Breakfast, lunch, dinner
                meal_spike = np.random.normal(30, 10)
            elif hour in [9, 14, 20]:  # Post-meal
                meal_spike = np.random.normal(20, 8)
            
            glucose = (self.health_parameters['glucose']['fasting_base'] + 
                      meal_spike + 
                      np.random.normal(0, self.health_parameters['glucose']['std']))
            glucose = np.clip(glucose, *self.health_parameters['glucose']['range'])
            
            # Temperature with circadian rhythm
            circadian_temp = (self.health_parameters['temperature']['circadian_amplitude'] * 
                             np.sin(2 * np.pi * (hour - 6) / 24))
            temperature = (self.health_parameters['temperature']['base'] + 
                          circadian_temp + 
                          np.random.normal(0, self.health_parameters['temperature']['std']))
            temperature = np.clip(temperature, *self.health_parameters['temperature']['range'])
            
            vitals_data.append({
                'timestamp': timestamp,
                'heart_rate': round(heart_rate),
                'systolic_bp': round(systolic_bp),
                'diastolic_bp': round(diastolic_bp),
                'glucose': round(glucose),
                'temperature': round(temperature, 1)
            })
        
        return vitals_data
    
    def generate_extended_sample_data(self, days=30, hours_per_day=24):
        """Generate extended health data for multiple days"""
        total_points = days * hours_per_day
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        timestamps = pd.date_range(start=start_time, end=end_time, periods=total_points)
        
        # Initialize base values
        base_hr = self.health_parameters['heart_rate']['rest_base']
        base_sys_bp = self.health_parameters['systolic_bp']['base']
        base_dia_bp = self.health_parameters['diastolic_bp']['base']
        base_glucose = self.health_parameters['glucose']['fasting_base']
        base_temp = self.health_parameters['temperature']['base']
        
        # Generate long-term trends
        hr_trend = np.linspace(0, 2, total_points)  # Slight increase over time
        bp_trend = np.linspace(0, 3, total_points)  # Slight BP increase
        glucose_trend = np.linspace(0, 5, total_points)  # Slight glucose increase
        
        extended_data = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Weekly patterns (higher stress on weekdays)
            weekly_stress = 1.2 if day_of_week < 5 else 0.8
            
            # Daily circadian patterns
            circadian_hr = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
            circadian_bp = 6 * np.sin(2 * np.pi * (hour - 8) / 24)
            circadian_temp = 1.0 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Heart rate
            heart_rate = (base_hr + hr_trend[i] + circadian_hr * weekly_stress + 
                         np.random.normal(0, 5))
            heart_rate = np.clip(heart_rate, 50, 120)
            
            # Blood pressure
            hr_influence = (heart_rate - base_hr) * 0.2
            systolic_bp = (base_sys_bp + bp_trend[i] + circadian_bp + hr_influence + 
                          np.random.normal(0, 8))
            systolic_bp = np.clip(systolic_bp, 90, 180)
            
            diastolic_bp = (base_dia_bp + bp_trend[i] * 0.7 + circadian_bp * 0.6 + 
                           hr_influence * 0.5 + np.random.normal(0, 6))
            diastolic_bp = np.clip(diastolic_bp, 60, 110)
            
            # Glucose with meal patterns
            meal_effect = 0
            if hour in [7, 8, 9]:  # Breakfast period
                meal_effect = np.random.uniform(10, 40)
            elif hour in [12, 13, 14]:  # Lunch period
                meal_effect = np.random.uniform(15, 45)
            elif hour in [18, 19, 20]:  # Dinner period
                meal_effect = np.random.uniform(20, 50)
            
            glucose = (base_glucose + glucose_trend[i] + meal_effect + 
                      np.random.normal(0, 10))
            glucose = np.clip(glucose, 70, 250)
            
            # Temperature
            temperature = (base_temp + circadian_temp + 
                          np.random.normal(0, 0.4))
            temperature = np.clip(temperature, 97.0, 100.0)
            
            extended_data.append({
                'timestamp': timestamp,
                'heart_rate': round(heart_rate),
                'systolic_bp': round(systolic_bp),
                'diastolic_bp': round(diastolic_bp),
                'glucose': round(glucose),
                'temperature': round(temperature, 1)
            })
        
        return extended_data
    
    def generate_user_profile(self, user_type='normal'):
        """Generate a realistic user profile"""
        user_profiles = {
            'young_healthy': {
                'age_range': (18, 35),
                'bmi_range': (18.5, 25),
                'activity_level': 'high',
                'health_conditions': []
            },
            'middle_aged': {
                'age_range': (35, 55),
                'bmi_range': (22, 28),
                'activity_level': 'moderate',
                'health_conditions': ['mild_hypertension']
            },
            'senior': {
                'age_range': (55, 80),
                'bmi_range': (24, 30),
                'activity_level': 'low',
                'health_conditions': ['hypertension', 'diabetes_risk']
            },
            'athlete': {
                'age_range': (20, 40),
                'bmi_range': (18, 24),
                'activity_level': 'very_high',
                'health_conditions': []
            },
            'high_risk': {
                'age_range': (40, 70),
                'bmi_range': (28, 35),
                'activity_level': 'low',
                'health_conditions': ['diabetes', 'hypertension', 'high_cholesterol']
            }
        }
        
        if user_type not in user_profiles:
            user_type = 'normal'
            profile = {
                'age_range': (25, 65),
                'bmi_range': (20, 30),
                'activity_level': 'moderate',
                'health_conditions': []
            }
        else:
            profile = user_profiles[user_type]
        
        # Generate specific user data
        age = np.random.randint(*profile['age_range'])
        height = np.random.normal(170, 10)  # cm
        height = np.clip(height, 150, 200)
        
        # BMI-based weight calculation
        target_bmi = np.random.uniform(*profile['bmi_range'])
        weight = target_bmi * (height / 100) ** 2
        
        gender = np.random.choice(['Male', 'Female'])
        
        user_profile = {
            'age': int(age),
            'gender': gender,
            'height': round(height, 1),
            'weight': round(weight, 1),
            'bmi': round(weight / (height / 100) ** 2, 1),
            'activity_level': profile['activity_level'],
            'health_conditions': profile['health_conditions']
        }
        
        return user_profile
    
    def generate_health_data_with_conditions(self, user_profile, num_points=100):
        """Generate health data based on user profile and conditions"""
        conditions = user_profile.get('health_conditions', [])
        age = user_profile.get('age', 30)
        bmi = user_profile.get('bmi', 25)
        activity_level = user_profile.get('activity_level', 'moderate')
        
        # Adjust base parameters based on profile
        age_factor = (age - 30) / 10  # Normalize around age 30
        bmi_factor = (bmi - 25) / 5   # Normalize around BMI 25
        
        activity_multipliers = {
            'very_high': 0.8,
            'high': 0.9,
            'moderate': 1.0,
            'low': 1.1,
            'very_low': 1.2
        }
        activity_mult = activity_multipliers.get(activity_level, 1.0)
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_points)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)
        
        health_data = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            
            # Base values with adjustments
            base_hr = 72 + age_factor * 2 + bmi_factor * 3
            base_sys_bp = 120 + age_factor * 5 + bmi_factor * 4
            base_dia_bp = 80 + age_factor * 3 + bmi_factor * 2
            base_glucose = 95 + age_factor * 2 + bmi_factor * 5
            base_temp = 98.6
            
            # Apply activity level effects
            base_hr *= activity_mult
            
            # Apply condition effects
            condition_adjustments = {
                'hypertension': {'sys_bp': 20, 'dia_bp': 10, 'hr': 5},
                'diabetes': {'glucose': 40, 'hr': 8},
                'diabetes_risk': {'glucose': 20, 'hr': 4},
                'high_cholesterol': {'sys_bp': 10},
                'mild_hypertension': {'sys_bp': 10, 'dia_bp': 5}
            }
            
            sys_bp_adj = 0
            dia_bp_adj = 0
            hr_adj = 0
            glucose_adj = 0
            
            for condition in conditions:
                if condition in condition_adjustments:
                    adj = condition_adjustments[condition]
                    sys_bp_adj += adj.get('sys_bp', 0)
                    dia_bp_adj += adj.get('dia_bp', 0)
                    hr_adj += adj.get('hr', 0)
                    glucose_adj += adj.get('glucose', 0)
            
            # Add circadian rhythms
            circadian_hr = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            circadian_bp = 4 * np.sin(2 * np.pi * (hour - 8) / 24)
            circadian_temp = 0.8 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Generate final values
            heart_rate = base_hr + hr_adj + circadian_hr + np.random.normal(0, 6)
            heart_rate = np.clip(heart_rate, 50, 140)
            
            systolic_bp = base_sys_bp + sys_bp_adj + circadian_bp + np.random.normal(0, 8)
            systolic_bp = np.clip(systolic_bp, 90, 200)
            
            diastolic_bp = base_dia_bp + dia_bp_adj + circadian_bp * 0.6 + np.random.normal(0, 6)
            diastolic_bp = np.clip(diastolic_bp, 60, 120)
            
            # Glucose with meal effects
            meal_effect = 0
            if hour in [7, 8, 9, 12, 13, 14, 18, 19, 20]:
                meal_effect = np.random.uniform(10, 40)
            
            glucose = base_glucose + glucose_adj + meal_effect + np.random.normal(0, 12)
            glucose = np.clip(glucose, 70, 300)
            
            temperature = base_temp + circadian_temp + np.random.normal(0, 0.4)
            temperature = np.clip(temperature, 97.0, 102.0)
            
            health_data.append({
                'timestamp': timestamp,
                'heart_rate': round(heart_rate),
                'systolic_bp': round(systolic_bp),
                'diastolic_bp': round(diastolic_bp),
                'glucose': round(glucose),
                'temperature': round(temperature, 1)
            })
        
        return health_data
    
    def generate_anomalous_data_point(self, normal_vitals, anomaly_type='random'):
        """Generate a single anomalous data point"""
        anomalous_vitals = normal_vitals.copy()
        
        anomaly_types = {
            'high_bp_crisis': {
                'systolic_bp': (180, 220),
                'diastolic_bp': (110, 130),
                'heart_rate': (100, 130)
            },
            'hypoglycemia': {
                'glucose': (40, 65),
                'heart_rate': (90, 120)
            },
            'hyperglycemia': {
                'glucose': (250, 400),
                'heart_rate': (85, 110)
            },
            'fever': {
                'temperature': (100.5, 104.0),
                'heart_rate': (95, 130)
            },
            'tachycardia': {
                'heart_rate': (120, 160)
            },
            'bradycardia': {
                'heart_rate': (35, 55)
            }
        }
        
        if anomaly_type == 'random':
            anomaly_type = np.random.choice(list(anomaly_types.keys()))
        
        if anomaly_type in anomaly_types:
            anomaly_params = anomaly_types[anomaly_type]
            
            for vital, (min_val, max_val) in anomaly_params.items():
                if vital in anomalous_vitals:
                    anomalous_vitals[vital] = np.random.uniform(min_val, max_val)
                    if vital != 'temperature':
                        anomalous_vitals[vital] = round(anomalous_vitals[vital])
                    else:
                        anomalous_vitals[vital] = round(anomalous_vitals[vital], 1)
        
        return anomalous_vitals, anomaly_type
    
    def generate_test_dataset(self, num_users=100, points_per_user=50):
        """Generate a comprehensive test dataset"""
        user_types = ['young_healthy', 'middle_aged', 'senior', 'athlete', 'high_risk']
        
        dataset = {
            'users': [],
            'health_data': [],
            'anomalies': []
        }
        
        for i in range(num_users):
            # Generate user profile
            user_type = np.random.choice(user_types)
            user_profile = self.generate_user_profile(user_type)
            user_profile['user_id'] = f"user_{i:03d}"
            user_profile['user_type'] = user_type
            
            # Generate health data
            health_data = self.generate_health_data_with_conditions(
                user_profile, points_per_user
            )
            
            # Add user_id to each health data point
            for data_point in health_data:
                data_point['user_id'] = user_profile['user_id']
            
            # Generate some anomalies (10% chance per data point)
            for j, data_point in enumerate(health_data):
                if np.random.random() < 0.1:  # 10% chance of anomaly
                    anomalous_point, anomaly_type = self.generate_anomalous_data_point(
                        data_point
                    )
                    anomalous_point['anomaly_type'] = anomaly_type
                    dataset['anomalies'].append(anomalous_point)
            
            dataset['users'].append(user_profile)
            dataset['health_data'].extend(health_data)
        
        return dataset
    
    def save_dataset_to_files(self, dataset, base_filename='health_dataset'):
        """Save dataset to CSV files"""
        # Save users
        users_df = pd.DataFrame(dataset['users'])
        users_df.to_csv(f"{base_filename}_users.csv", index=False)
        
        # Save health data
        health_df = pd.DataFrame(dataset['health_data'])
        health_df.to_csv(f"{base_filename}_health_data.csv", index=False)
        
        # Save anomalies
        if dataset['anomalies']:
            anomalies_df = pd.DataFrame(dataset['anomalies'])
            anomalies_df.to_csv(f"{base_filename}_anomalies.csv", index=False)
        
        print(f"Dataset saved to {base_filename}_*.csv files")
        print(f"Users: {len(dataset['users'])}")
        print(f"Health data points: {len(dataset['health_data'])}")
        print(f"Anomalies: {len(dataset['anomalies'])}")

# Example usage
if __name__ == "__main__":
    # Initialize data generator
    generator = HealthDataGenerator()
    
    # Generate sample vitals for demo
    sample_vitals = generator.generate_sample_vitals(24)
    print("Sample vitals for last 24 hours:")
    for vital in sample_vitals[-3:]:  # Show last 3
        print(f"Time: {vital['timestamp'].strftime('%H:%M')}, "
              f"HR: {vital['heart_rate']}, "
              f"BP: {vital['systolic_bp']}/{vital['diastolic_bp']}, "
              f"Glucose: {vital['glucose']}")
    
    # Generate user profile
    user_profile = generator.generate_user_profile('middle_aged')
    print(f"\nSample user profile: {user_profile}")
    
    # Generate health data for user
    health_data = generator.generate_health_data_with_conditions(user_profile, 20)
    print(f"\nGenerated {len(health_data)} health data points for user")
    
    # Generate anomalous data point
    normal_point = health_data[0]
    anomalous_point, anomaly_type = generator.generate_anomalous_data_point(
        normal_point, 'high_bp_crisis'
    )
    print(f"\nAnomaly generated ({anomaly_type}):")
    print(f"Normal: HR={normal_point['heart_rate']}, BP={normal_point['systolic_bp']}/{normal_point['diastolic_bp']}")
    print(f"Anomalous: HR={anomalous_point['heart_rate']}, BP={anomalous_point['systolic_bp']}/{anomalous_point['diastolic_bp']}")