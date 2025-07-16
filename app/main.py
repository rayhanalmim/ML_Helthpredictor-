import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.disease_prediction import DiseasePredictionModel
from models.clustering import HealthClustering
from models.time_series import VitalSignsPredictor
from models.anomaly_detection import AnomalyDetector
from models.recommendation_engine import HealthRecommendationEngine
from chatbot.symptom_checker import SymptomChecker
from data_processing.data_generator import HealthDataGenerator
from utils.database import DatabaseManager
from utils.visualizations import HealthVisualizations

# Page configuration
st.set_page_config(
    page_title="AI Health Monitoring System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e88e5;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffecb5;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'health_history' not in st.session_state:
    st.session_state.health_history = []

class HealthMonitoringApp:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_generator = HealthDataGenerator()
        self.visualizations = HealthVisualizations()
        
        # Initialize ML models
        self.disease_predictor = DiseasePredictionModel()
        self.clustering_model = HealthClustering()
        self.time_series_predictor = VitalSignsPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.recommendation_engine = HealthRecommendationEngine()
        self.symptom_checker = SymptomChecker()
        
    def main(self):
        # Sidebar
        self.create_sidebar()
        
        # Main content
        st.markdown('<h1 class="main-header">🏥 AI Health Monitoring & Disease Prediction System</h1>', 
                   unsafe_allow_html=True)
        
        # Navigation
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "🔍 Health Assessment", 
            "💬 Symptom Checker", 
            "📈 Analytics", 
            "🎯 Recommendations", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_dashboard()
            
        with tab2:
            self.render_health_assessment()
            
        with tab3:
            self.render_symptom_checker()
            
        with tab4:
            self.render_analytics()
            
        with tab5:
            self.render_recommendations()
            
        with tab6:
            self.render_settings()
    
    def create_sidebar(self):
        st.sidebar.title("🏥 Health Monitor")
        
        # User profile section
        st.sidebar.subheader("👤 User Profile")
        
        with st.sidebar.form("user_profile"):
            name = st.text_input("Name", value=st.session_state.user_data.get('name', ''))
            age = st.number_input("Age", min_value=1, max_value=120, 
                                value=st.session_state.user_data.get('age', 30))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                index=["Male", "Female", "Other"].index(
                                    st.session_state.user_data.get('gender', 'Male')))
            height = st.number_input("Height (cm)", min_value=50, max_value=250,
                                   value=st.session_state.user_data.get('height', 170))
            weight = st.number_input("Weight (kg)", min_value=20, max_value=300,
                                   value=st.session_state.user_data.get('weight', 70))
            
            submit_profile = st.form_submit_button("Update Profile")
            
            if submit_profile:
                st.session_state.user_data.update({
                    'name': name, 'age': age, 'gender': gender,
                    'height': height, 'weight': weight,
                    'bmi': weight / ((height/100) ** 2)
                })
                st.success("Profile updated!")
        
        # Quick health input
        st.sidebar.subheader("⚡ Quick Health Input")
        
        with st.sidebar.form("quick_vitals"):
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            blood_pressure_sys = st.number_input("Systolic BP", min_value=70, max_value=250, value=120)
            blood_pressure_dia = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
            glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=500, value=100)
            temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
            
            submit_vitals = st.form_submit_button("Add Vitals")
            
            if submit_vitals:
                vital_data = {
                    'timestamp': datetime.now(),
                    'heart_rate': heart_rate,
                    'systolic_bp': blood_pressure_sys,
                    'diastolic_bp': blood_pressure_dia,
                    'glucose': glucose,
                    'temperature': temperature
                }
                st.session_state.health_history.append(vital_data)
                st.success("Vitals recorded!")
    
    def render_dashboard(self):
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate sample data if no history exists
        if not st.session_state.health_history:
            sample_data = self.data_generator.generate_sample_vitals()
            st.session_state.health_history = sample_data
        
        recent_vitals = st.session_state.health_history[-1] if st.session_state.health_history else {}
        
        with col1:
            st.metric(
                label="❤️ Heart Rate",
                value=f"{recent_vitals.get('heart_rate', 72)} bpm",
                delta=f"{np.random.randint(-3, 4)} bpm"
            )
        
        with col2:
            st.metric(
                label="🩸 Blood Pressure",
                value=f"{recent_vitals.get('systolic_bp', 120)}/{recent_vitals.get('diastolic_bp', 80)}",
                delta=f"{np.random.randint(-2, 3)} mmHg"
            )
        
        with col3:
            st.metric(
                label="🍯 Glucose",
                value=f"{recent_vitals.get('glucose', 100)} mg/dL",
                delta=f"{np.random.randint(-5, 6)} mg/dL"
            )
        
        with col4:
            st.metric(
                label="🌡️ Temperature",
                value=f"{recent_vitals.get('temperature', 98.6)}°F",
                delta=f"{np.random.uniform(-0.5, 0.5):.1f}°F"
            )
        
        # Health Status Overview
        st.subheader("📊 Health Status Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Vital signs chart
            if st.session_state.health_history:
                df = pd.DataFrame(st.session_state.health_history)
                fig = self.visualizations.create_vitals_timeline(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Health risk assessment
            risk_score = self.calculate_health_risk()
            self.display_risk_assessment(risk_score)
        
        # Anomaly detection alerts
        self.display_anomaly_alerts()
    
    def render_health_assessment(self):
        st.subheader("🔍 Comprehensive Health Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩺 Disease Risk Prediction")
            
            # Input health parameters
            with st.form("disease_prediction"):
                chest_pain = st.selectbox("Chest Pain Type", 
                                        ["No Pain", "Typical Angina", "Atypical Angina", "Non-anginal"])
                cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
                max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
                exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                family_history = st.multiselect("Family History", 
                                               ["Heart Disease", "Diabetes", "Hypertension", "Cancer"])
                
                predict_button = st.form_submit_button("Predict Disease Risk")
                
                if predict_button:
                    # Prepare data for prediction
                    user_data = st.session_state.user_data
                    prediction_data = {
                        'age': user_data.get('age', 30),
                        'gender': 1 if user_data.get('gender') == 'Male' else 0,
                        'chest_pain': ["No Pain", "Typical Angina", "Atypical Angina", "Non-anginal"].index(chest_pain),
                        'cholesterol': cholesterol,
                        'max_heart_rate': max_heart_rate,
                        'exercise_angina': 1 if exercise_angina == "Yes" else 0
                    }
                    
                    # Get predictions
                    predictions = self.disease_predictor.predict_multiple_diseases(prediction_data)
                    
                    st.subheader("🎯 Disease Risk Results")
                    for disease, risk in predictions.items():
                        color = "🔴" if risk > 0.7 else "🟡" if risk > 0.4 else "🟢"
                        st.write(f"{color} **{disease}**: {risk:.1%} risk")
        
        with col2:
            st.subheader("👥 Health Profile Clustering")
            
            # Clustering analysis
            if st.button("Analyze Health Profile"):
                user_data = st.session_state.user_data
                if user_data:
                    cluster_data = {
                        'age': user_data.get('age', 30),
                        'bmi': user_data.get('bmi', 25),
                        'heart_rate': st.session_state.health_history[-1].get('heart_rate', 72) if st.session_state.health_history else 72,
                        'systolic_bp': st.session_state.health_history[-1].get('systolic_bp', 120) if st.session_state.health_history else 120
                    }
                    
                    cluster_result = self.clustering_model.predict_cluster(cluster_data)
                    
                    st.write(f"**Your Health Profile**: Cluster {cluster_result['cluster']}")
                    st.write(f"**Profile Description**: {cluster_result['description']}")
                    st.write(f"**Similar Users**: {cluster_result['similar_users']} people")
                    
                    # Display cluster characteristics
                    characteristics = cluster_result.get('characteristics', [])
                    if characteristics:
                        st.write("**Common Characteristics:**")
                        for char in characteristics:
                            st.write(f"- {char}")
    
    def render_symptom_checker(self):
        st.subheader("💬 AI-Powered Symptom Checker")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Describe your symptoms in natural language, and our AI will help analyze them.")
            
            # Symptom input
            symptoms_text = st.text_area(
                "Describe your symptoms:",
                placeholder="e.g., I have been experiencing headaches, fatigue, and dizziness for the past 3 days...",
                height=150
            )
            
            # Quick symptom selection
            st.subheader("🎯 Quick Symptom Selection")
            common_symptoms = [
                "Headache", "Fever", "Cough", "Fatigue", "Nausea", "Dizziness",
                "Chest Pain", "Shortness of Breath", "Joint Pain", "Muscle Aches",
                "Sore Throat", "Runny Nose", "Loss of Appetite", "Sleep Issues"
            ]
            
            selected_symptoms = st.multiselect("Select symptoms:", common_symptoms)
            
            if st.button("🔍 Analyze Symptoms", type="primary"):
                if symptoms_text or selected_symptoms:
                    # Combine text and selected symptoms
                    full_symptom_description = symptoms_text
                    if selected_symptoms:
                        full_symptom_description += " " + ", ".join(selected_symptoms)
                    
                    # Get AI analysis
                    analysis = self.symptom_checker.analyze_symptoms(full_symptom_description)
                    
                    st.subheader("🩺 Analysis Results")
                    
                    # Display possible conditions
                    st.write("**Possible Conditions:**")
                    for condition in analysis.get('possible_conditions', []):
                        confidence = condition.get('confidence', 0)
                        color = "🔴" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🟢"
                        st.write(f"{color} **{condition['name']}** - {confidence:.1%} match")
                        st.write(f"   *{condition.get('description', '')}*")
                    
                    # Display recommendations
                    st.write("**Recommendations:**")
                    for rec in analysis.get('recommendations', []):
                        st.write(f"- {rec}")
                    
                    # Urgency assessment
                    urgency = analysis.get('urgency', 'low')
                    if urgency == 'high':
                        st.error("⚠️ **Urgent**: Please seek immediate medical attention!")
                    elif urgency == 'medium':
                        st.warning("⚡ **Moderate**: Consider consulting a healthcare provider.")
                    else:
                        st.info("ℹ️ **Low**: Monitor symptoms and consider rest.")
        
        with col2:
            # Symptom history
            st.subheader("📅 Symptom History")
            if 'symptom_history' not in st.session_state:
                st.session_state.symptom_history = []
            
            if st.session_state.symptom_history:
                for i, entry in enumerate(st.session_state.symptom_history[-5:]):
                    with st.expander(f"Entry {len(st.session_state.symptom_history) - i}"):
                        st.write(f"**Date**: {entry['date']}")
                        st.write(f"**Symptoms**: {entry['symptoms'][:100]}...")
                        st.write(f"**Top Condition**: {entry['top_condition']}")
            else:
                st.write("No symptom history yet.")
    
    def render_analytics(self):
        st.subheader("📈 Advanced Health Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📊 Trends", "🔮 Predictions", "⚠️ Anomalies"])
        
        with tab1:
            self.render_health_trends()
        
        with tab2:
            self.render_predictions()
        
        with tab3:
            self.render_anomaly_detection()
    
    def render_health_trends(self):
        if not st.session_state.health_history:
            st.info("No health data available. Please add some vitals first.")
            return
        
        df = pd.DataFrame(st.session_state.health_history)
        
        # Time series visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Heart rate trend
            fig = px.line(df, x='timestamp', y='heart_rate', 
                         title='❤️ Heart Rate Trend',
                         labels={'heart_rate': 'Heart Rate (bpm)', 'timestamp': 'Time'})
            fig.update_traces(line_color='#e74c3c')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Blood pressure trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['systolic_bp'], 
                                   mode='lines+markers', name='Systolic', line_color='#3498db'))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['diastolic_bp'], 
                                   mode='lines+markers', name='Diastolic', line_color='#e67e22'))
            fig.update_layout(title='🩸 Blood Pressure Trend', 
                            xaxis_title='Time', yaxis_title='Blood Pressure (mmHg)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        summary_stats = df[['heart_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'temperature']].describe()
        st.dataframe(summary_stats)
    
    def render_predictions(self):
        st.subheader("🔮 Future Health Predictions")
        
        if len(st.session_state.health_history) < 5:
            st.warning("Need at least 5 data points for reliable predictions.")
            return
        
        df = pd.DataFrame(st.session_state.health_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Heart rate prediction
            if st.button("Predict Heart Rate Trend"):
                predictions = self.time_series_predictor.predict_heart_rate(df)
                
                fig = go.Figure()
                # Historical data
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate'], 
                                       mode='lines+markers', name='Historical', line_color='#3498db'))
                # Predictions
                future_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=8, freq='H')[1:]
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, 
                                       mode='lines+markers', name='Predicted', line_color='#e74c3c'))
                
                fig.update_layout(title='❤️ Heart Rate Prediction (Next 7 Hours)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Blood pressure prediction
            if st.button("Predict Blood Pressure"):
                bp_predictions = self.time_series_predictor.predict_blood_pressure(df)
                
                fig = go.Figure()
                future_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=8, freq='H')[1:]
                
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['systolic_bp'], 
                                       mode='lines+markers', name='Historical Systolic', line_color='#3498db'))
                fig.add_trace(go.Scatter(x=future_dates, y=bp_predictions['systolic'], 
                                       mode='lines+markers', name='Predicted Systolic', line_color='#e74c3c'))
                
                fig.update_layout(title='🩸 Blood Pressure Prediction')
                st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_detection(self):
        if len(st.session_state.health_history) < 10:
            st.warning("Need at least 10 data points for anomaly detection.")
            return
        
        df = pd.DataFrame(st.session_state.health_history)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(df)
        
        if anomalies['has_anomalies']:
            st.error("⚠️ **Anomalies Detected!**")
            
            for anomaly in anomalies['anomaly_details']:
                # Display affected vitals and severity
                if 'affected_vitals' in anomaly and anomaly['affected_vitals']:
                    for vital in anomaly['affected_vitals'][:2]:  # Show up to 2 vitals
                        st.warning(f"**{vital}**: {anomaly['severity']} severity")
                    
                    # Display recommendations if available
                    if 'recommendations' in anomaly and anomaly['recommendations']:
                        st.write("**Recommendations:**")
                        for rec in anomaly['recommendations'][:3]:  # Show top 3 recommendations
                            st.write(f"- {rec}")
                else:
                    st.warning(f"**Anomaly detected**: {anomaly.get('severity', 'unknown')} severity")
        else:
            st.success("✅ No anomalies detected in your health data.")
        
        # Visualize anomalies
        fig = self.visualizations.create_anomaly_plot(df, anomalies)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self):
        st.subheader("🎯 Personalized Health Recommendations")
        
        user_data = st.session_state.user_data
        health_data = st.session_state.health_history
        
        if not user_data or not health_data:
            st.info("Please complete your profile and add health data to get personalized recommendations.")
            return
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(user_data, health_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💪 Fitness Recommendations")
            for rec in recommendations.get('fitness', []):
                st.write(f"- {rec}")
        
        with col2:
            st.subheader("🥗 Nutrition Recommendations")
            for rec in recommendations.get('nutrition', []):
                st.write(f"- {rec}")
        
        # Lifestyle recommendations
        st.subheader("🌱 Lifestyle Recommendations")
        for rec in recommendations.get('lifestyle', []):
            st.write(f"- {rec}")
        
        # Medication reminders (if any)
        if recommendations.get('medications'):
            st.subheader("💊 Medication Reminders")
            for med in recommendations['medications']:
                st.write(f"- {med}")
    
    def render_settings(self):
        st.subheader("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔔 Alert Settings")
            
            heart_rate_high = st.number_input("Heart Rate High Alert", value=100)
            heart_rate_low = st.number_input("Heart Rate Low Alert", value=60)
            bp_high = st.number_input("Blood Pressure High Alert", value=140)
            glucose_high = st.number_input("Glucose High Alert", value=140)
            
            if st.button("Save Alert Settings"):
                st.session_state.alert_settings = {
                    'heart_rate_high': heart_rate_high,
                    'heart_rate_low': heart_rate_low,
                    'bp_high': bp_high,
                    'glucose_high': glucose_high
                }
                st.success("Alert settings saved!")
        
        with col2:
            st.subheader("📊 Data Management")
            
            if st.button("Export Health Data"):
                if st.session_state.health_history:
                    df = pd.DataFrame(st.session_state.health_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No health data to export.")
            
            if st.button("Clear All Data"):
                if st.button("Confirm Clear Data"):
                    st.session_state.health_history = []
                    st.session_state.user_data = {}
                    st.success("All data cleared!")
            
            if st.button("Generate Sample Data"):
                sample_data = self.data_generator.generate_extended_sample_data()
                st.session_state.health_history = sample_data
                st.success("Sample data generated!")
    
    def calculate_health_risk(self):
        """Calculate overall health risk score"""
        if not st.session_state.health_history or not st.session_state.user_data:
            return 0.5
        
        recent_vitals = st.session_state.health_history[-1]
        user_data = st.session_state.user_data
        
        risk_factors = 0
        total_factors = 0
        
        # Age factor
        age = user_data.get('age', 30)
        if age > 65:
            risk_factors += 2
        elif age > 45:
            risk_factors += 1
        total_factors += 2
        
        # BMI factor
        bmi = user_data.get('bmi', 25)
        if bmi > 30:
            risk_factors += 2
        elif bmi > 25:
            risk_factors += 1
        total_factors += 2
        
        # Heart rate factor
        hr = recent_vitals.get('heart_rate', 72)
        if hr > 100 or hr < 60:
            risk_factors += 1
        total_factors += 1
        
        # Blood pressure factor
        sys_bp = recent_vitals.get('systolic_bp', 120)
        if sys_bp > 140:
            risk_factors += 2
        elif sys_bp > 120:
            risk_factors += 1
        total_factors += 2
        
        return risk_factors / total_factors if total_factors > 0 else 0.5
    
    def display_risk_assessment(self, risk_score):
        """Display health risk assessment"""
        if risk_score < 0.3:
            st.success("🟢 **Low Risk** - Your health indicators are within normal ranges.")
        elif risk_score < 0.6:
            st.warning("🟡 **Moderate Risk** - Some health indicators need attention.")
        else:
            st.error("🔴 **High Risk** - Multiple health indicators are concerning. Consider consulting a healthcare provider.")
        
        # Risk score visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_anomaly_alerts(self):
        """Display any anomaly alerts"""
        if len(st.session_state.health_history) >= 5:
            df = pd.DataFrame(st.session_state.health_history)
            anomalies = self.anomaly_detector.detect_anomalies(df)
            
            if anomalies['has_anomalies']:
                st.error("⚠️ **Health Anomalies Detected!**")
                for anomaly in anomalies['anomaly_details'][:3]:  # Show top 3
                    # Display the affected vitals and severity
                    if anomaly['affected_vitals']:
                        for vital in anomaly['affected_vitals'][:2]:  # Show up to 2 vitals
                            st.warning(f"- {vital}: {anomaly['severity']} severity")
                    else:
                        st.warning(f"- Anomaly detected: {anomaly['severity']} severity")

# Run the application
if __name__ == "__main__":
    app = HealthMonitoringApp()
    app.main()