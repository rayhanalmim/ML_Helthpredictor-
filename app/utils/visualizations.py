import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class HealthVisualizations:
    def __init__(self):
        self.color_palette = {
            'heart_rate': '#e74c3c',
            'blood_pressure': '#3498db',
            'glucose': '#f39c12',
            'temperature': '#9b59b6',
            'normal': '#2ecc71',
            'warning': '#f1c40f',
            'danger': '#e74c3c'
        }
    
    def create_vitals_timeline(self, data):
        """Create timeline chart of vital signs"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Heart Rate', 'Blood Pressure', 'Glucose', 'Temperature'),
            vertical_spacing=0.12
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], 
                y=data['heart_rate'],
                mode='lines+markers',
                name='Heart Rate',
                line=dict(color=self.color_palette['heart_rate'])
            ),
            row=1, col=1
        )
        
        # Blood Pressure
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], 
                y=data['systolic_bp'],
                mode='lines+markers',
                name='Systolic',
                line=dict(color=self.color_palette['blood_pressure'])
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], 
                y=data['diastolic_bp'],
                mode='lines+markers',
                name='Diastolic',
                line=dict(color='#85C1E9')
            ),
            row=1, col=2
        )
        
        # Glucose
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], 
                y=data['glucose'],
                mode='lines+markers',
                name='Glucose',
                line=dict(color=self.color_palette['glucose'])
            ),
            row=2, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'], 
                y=data['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color=self.color_palette['temperature'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Vital Signs Timeline",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_anomaly_plot(self, data, anomalies):
        """Create plot highlighting anomalies"""
        fig = go.Figure()
        
        # Normal data
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['heart_rate'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color=self.color_palette['normal'])
        ))
        
        # Highlight anomalies if any
        if anomalies['has_anomalies']:
            # This is a simplified version - in practice you'd extract anomaly points
            fig.add_trace(go.Scatter(
                x=[data['timestamp'].iloc[-1]],
                y=[data['heart_rate'].iloc[-1]],
                mode='markers',
                name='Anomaly',
                marker=dict(color=self.color_palette['danger'], size=12)
            ))
        
        fig.update_layout(
            title="Health Data with Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Heart Rate (bpm)"
        )
        
        return fig
    
    def create_health_dashboard(self, user_data, health_data):
        """Create comprehensive health dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Heart Rate Trend', 'Blood Pressure',
                'Glucose Levels', 'Temperature',
                'Health Score', 'Risk Assessment'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        if not health_data.empty:
            # Heart Rate
            fig.add_trace(
                go.Scatter(x=health_data['timestamp'], y=health_data['heart_rate'],
                          mode='lines', name='Heart Rate', 
                          line=dict(color=self.color_palette['heart_rate'])),
                row=1, col=1
            )
            
            # Blood Pressure
            fig.add_trace(
                go.Scatter(x=health_data['timestamp'], y=health_data['systolic_bp'],
                          mode='lines', name='Systolic', 
                          line=dict(color=self.color_palette['blood_pressure'])),
                row=1, col=2
            )
            
            # Glucose
            fig.add_trace(
                go.Scatter(x=health_data['timestamp'], y=health_data['glucose'],
                          mode='lines', name='Glucose',
                          line=dict(color=self.color_palette['glucose'])),
                row=2, col=1
            )
            
            # Temperature
            fig.add_trace(
                go.Scatter(x=health_data['timestamp'], y=health_data['temperature'],
                          mode='lines', name='Temperature',
                          line=dict(color=self.color_palette['temperature'])),
                row=2, col=2
            )
        
        # Health Score Indicator
        health_score = self._calculate_health_score(health_data)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "green"}],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 90}}
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, title="Health Dashboard")
        return fig
    
    def _calculate_health_score(self, health_data):
        """Calculate overall health score"""
        if health_data.empty:
            return 50
        
        # Simple scoring based on recent averages
        recent_data = health_data.tail(10)
        
        score = 100
        
        # Heart rate scoring
        avg_hr = recent_data['heart_rate'].mean()
        if avg_hr < 60 or avg_hr > 100:
            score -= 20
        
        # Blood pressure scoring
        avg_sys_bp = recent_data['systolic_bp'].mean()
        if avg_sys_bp > 140:
            score -= 30
        elif avg_sys_bp > 120:
            score -= 10
        
        # Glucose scoring
        avg_glucose = recent_data['glucose'].mean()
        if avg_glucose > 140:
            score -= 25
        elif avg_glucose > 100:
            score -= 10
        
        return max(0, score)

# Example usage
if __name__ == "__main__":
    viz = HealthVisualizations()
    
    # Sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
        'heart_rate': np.random.normal(75, 10, 24),
        'systolic_bp': np.random.normal(120, 15, 24),
        'diastolic_bp': np.random.normal(80, 10, 24),
        'glucose': np.random.normal(100, 20, 24),
        'temperature': np.random.normal(98.6, 1, 24)
    })
    
    fig = viz.create_vitals_timeline(sample_data)
    print("Visualization created successfully")