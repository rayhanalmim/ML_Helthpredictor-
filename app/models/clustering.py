import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HealthClustering:
    def __init__(self):
        self.kmeans_model = None
        self.hierarchical_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.is_fitted = False
        self.cluster_profiles = {}
        self.feature_importance = {}
        
        # Generate synthetic health data for clustering
        self._generate_clustering_data()
        self._fit_clustering_models()
    
    def _generate_clustering_data(self):
        """Generate synthetic health data for clustering analysis"""
        np.random.seed(42)
        n_samples = 2000
        
        # Generate health profiles for different user types
        
        # Healthy Young Adults (Cluster 1)
        healthy_young = pd.DataFrame({
            'age': np.random.normal(25, 5, 400),
            'bmi': np.random.normal(22, 2, 400),
            'heart_rate': np.random.normal(70, 8, 400),
            'systolic_bp': np.random.normal(115, 8, 400),
            'diastolic_bp': np.random.normal(75, 5, 400),
            'cholesterol': np.random.normal(180, 20, 400),
            'glucose': np.random.normal(90, 10, 400),
            'exercise_frequency': np.random.normal(4, 1, 400),
            'sleep_hours': np.random.normal(8, 1, 400),
            'stress_level': np.random.normal(3, 1, 400)
        })
        
        # Middle-aged Moderate Risk (Cluster 2)
        middle_aged = pd.DataFrame({
            'age': np.random.normal(45, 8, 500),
            'bmi': np.random.normal(26, 3, 500),
            'heart_rate': np.random.normal(75, 10, 500),
            'systolic_bp': np.random.normal(125, 12, 500),
            'diastolic_bp': np.random.normal(80, 8, 500),
            'cholesterol': np.random.normal(210, 25, 500),
            'glucose': np.random.normal(105, 15, 500),
            'exercise_frequency': np.random.normal(2.5, 1, 500),
            'sleep_hours': np.random.normal(7, 1, 500),
            'stress_level': np.random.normal(5, 1.5, 500)
        })
        
        # Seniors with Health Issues (Cluster 3)
        seniors = pd.DataFrame({
            'age': np.random.normal(68, 8, 400),
            'bmi': np.random.normal(28, 4, 400),
            'heart_rate': np.random.normal(72, 12, 400),
            'systolic_bp': np.random.normal(140, 20, 400),
            'diastolic_bp': np.random.normal(85, 10, 400),
            'cholesterol': np.random.normal(240, 30, 400),
            'glucose': np.random.normal(120, 25, 400),
            'exercise_frequency': np.random.normal(1.5, 0.8, 400),
            'sleep_hours': np.random.normal(6.5, 1, 400),
            'stress_level': np.random.normal(4, 1, 400)
        })
        
        # Athletic/Fitness Enthusiasts (Cluster 4)
        athletes = pd.DataFrame({
            'age': np.random.normal(30, 8, 300),
            'bmi': np.random.normal(21, 2, 300),
            'heart_rate': np.random.normal(60, 8, 300),
            'systolic_bp': np.random.normal(110, 8, 300),
            'diastolic_bp': np.random.normal(70, 5, 300),
            'cholesterol': np.random.normal(170, 15, 300),
            'glucose': np.random.normal(85, 8, 300),
            'exercise_frequency': np.random.normal(6, 1, 300),
            'sleep_hours': np.random.normal(8.5, 0.8, 300),
            'stress_level': np.random.normal(2, 0.8, 300)
        })
        
        # High-Risk Individuals (Cluster 5)
        high_risk = pd.DataFrame({
            'age': np.random.normal(55, 10, 400),
            'bmi': np.random.normal(32, 4, 400),
            'heart_rate': np.random.normal(85, 15, 400),
            'systolic_bp': np.random.normal(155, 25, 400),
            'diastolic_bp': np.random.normal(95, 12, 400),
            'cholesterol': np.random.normal(280, 40, 400),
            'glucose': np.random.normal(140, 30, 400),
            'exercise_frequency': np.random.normal(1, 0.5, 400),
            'sleep_hours': np.random.normal(6, 1.2, 400),
            'stress_level': np.random.normal(7, 1.5, 400)
        })
        
        # Combine all data
        self.health_data = pd.concat([
            healthy_young, middle_aged, seniors, athletes, high_risk
        ], ignore_index=True)
        
        # Add some noise and ensure realistic ranges
        self.health_data = self.health_data.clip(lower={
            'age': 18, 'bmi': 15, 'heart_rate': 50, 'systolic_bp': 80,
            'diastolic_bp': 50, 'cholesterol': 120, 'glucose': 70,
            'exercise_frequency': 0, 'sleep_hours': 4, 'stress_level': 1
        })
        
        self.health_data = self.health_data.clip(upper={
            'age': 90, 'bmi': 50, 'heart_rate': 120, 'systolic_bp': 200,
            'diastolic_bp': 120, 'cholesterol': 400, 'glucose': 300,
            'exercise_frequency': 7, 'sleep_hours': 12, 'stress_level': 10
        })
        
        # Add derived features
        self.health_data['pulse_pressure'] = (
            self.health_data['systolic_bp'] - self.health_data['diastolic_bp']
        )
        self.health_data['health_score'] = (
            5 - (self.health_data['bmi'] - 22).abs() / 5 +
            (8 - self.health_data['stress_level']) +
            self.health_data['exercise_frequency'] +
            (self.health_data['sleep_hours'] - 4) / 2
        ).clip(0, 20)
    
    def _fit_clustering_models(self):
        """Fit different clustering models"""
        # Prepare features for clustering
        features = ['age', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                   'cholesterol', 'glucose', 'exercise_frequency', 'sleep_hours',
                   'stress_level', 'pulse_pressure', 'health_score']
        
        X = self.health_data[features]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        optimal_k = self._find_optimal_clusters(X_scaled)
        
        # Fit K-Means
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Fit Hierarchical Clustering
        self.hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = self.hierarchical_model.fit_predict(X_scaled)
        
        # Fit DBSCAN
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = self.dbscan_model.fit_predict(X_scaled)
        
        # Store cluster labels
        self.health_data['kmeans_cluster'] = kmeans_labels
        self.health_data['hierarchical_cluster'] = hierarchical_labels
        self.health_data['dbscan_cluster'] = dbscan_labels
        
        # Fit PCA for visualization
        self.pca_features = self.pca.fit_transform(X_scaled)
        
        # Create cluster profiles
        self._create_cluster_profiles(features)
        
        # Calculate feature importance
        self._calculate_feature_importance(features, X_scaled, kmeans_labels)
        
        self.is_fitted = True
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"K-Means Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
        print(f"Hierarchical Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.3f}")
        
        # Count DBSCAN clusters (excluding noise points)
        unique_dbscan = np.unique(dbscan_labels)
        n_dbscan_clusters = len(unique_dbscan) - (1 if -1 in unique_dbscan else 0)
        print(f"DBSCAN found {n_dbscan_clusters} clusters")
    
    def _find_optimal_clusters(self, X_scaled, max_k=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Use elbow method - find the point where inertia reduction slows down
        # Simple heuristic: look for the largest decrease in inertia
        decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        optimal_k = k_range[np.argmax(decreases)]
        
        # Ensure we have at least 3 clusters but not more than 7
        optimal_k = max(3, min(optimal_k, 7))
        
        return optimal_k
    
    def _create_cluster_profiles(self, features):
        """Create profiles for each cluster"""
        for cluster_id in self.health_data['kmeans_cluster'].unique():
            cluster_data = self.health_data[self.health_data['kmeans_cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.health_data) * 100,
                'avg_age': cluster_data['age'].mean(),
                'avg_bmi': cluster_data['bmi'].mean(),
                'avg_heart_rate': cluster_data['heart_rate'].mean(),
                'avg_bp': cluster_data['systolic_bp'].mean(),
                'avg_health_score': cluster_data['health_score'].mean(),
                'characteristics': [],
                'risk_level': 'Unknown'
            }
            
            # Determine cluster characteristics
            if profile['avg_age'] < 35 and profile['avg_bmi'] < 25 and profile['avg_health_score'] > 15:
                profile['characteristics'] = [
                    "Young and healthy", "Low BMI", "Good exercise habits", 
                    "Low stress levels", "Excellent vital signs"
                ]
                profile['risk_level'] = 'Low'
                profile['description'] = "Healthy Young Adults"
                
            elif profile['avg_age'] > 60 and (profile['avg_bp'] > 140 or profile['avg_bmi'] > 28):
                profile['characteristics'] = [
                    "Senior age group", "Elevated blood pressure", "Higher BMI",
                    "May have chronic conditions", "Needs regular monitoring"
                ]
                profile['risk_level'] = 'High'
                profile['description'] = "Seniors with Health Concerns"
                
            elif profile['avg_health_score'] > 16 and cluster_data['exercise_frequency'].mean() > 5:
                profile['characteristics'] = [
                    "High fitness level", "Excellent exercise habits", "Low resting heart rate",
                    "Optimal BMI", "Low stress levels"
                ]
                profile['risk_level'] = 'Very Low'
                profile['description'] = "Fitness Enthusiasts"
                
            elif profile['avg_bmi'] > 30 or profile['avg_bp'] > 150:
                profile['characteristics'] = [
                    "High BMI or blood pressure", "Multiple risk factors",
                    "Low exercise frequency", "Higher stress levels"
                ]
                profile['risk_level'] = 'High'
                profile['description'] = "High-Risk Individuals"
                
            else:
                profile['characteristics'] = [
                    "Moderate health status", "Some risk factors present",
                    "Average fitness level", "Room for improvement"
                ]
                profile['risk_level'] = 'Moderate'
                profile['description'] = "Moderate Risk Group"
            
            self.cluster_profiles[cluster_id] = profile
    
    def _calculate_feature_importance(self, features, X_scaled, labels):
        """Calculate feature importance for clustering"""
        # Calculate variance within clusters vs between clusters
        feature_importance = {}
        
        for i, feature in enumerate(features):
            feature_values = X_scaled[:, i]
            total_variance = np.var(feature_values)
            
            within_cluster_variance = 0
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                if np.sum(cluster_mask) > 1:
                    within_cluster_variance += np.var(feature_values[cluster_mask]) * np.sum(cluster_mask)
            
            within_cluster_variance /= len(feature_values)
            between_cluster_variance = total_variance - within_cluster_variance
            
            # Higher ratio means feature is more important for clustering
            importance = between_cluster_variance / (within_cluster_variance + 1e-10)
            feature_importance[feature] = importance
        
        # Normalize importance scores
        max_importance = max(feature_importance.values())
        self.feature_importance = {
            k: v / max_importance for k, v in feature_importance.items()
        }
    
    def predict_cluster(self, user_data):
        """Predict cluster for a new user"""
        if not self.is_fitted:
            return {'cluster': 0, 'description': 'Model not fitted', 'similar_users': 0}
        
        # Prepare user data
        features = ['age', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                   'cholesterol', 'glucose', 'exercise_frequency', 'sleep_hours',
                   'stress_level']
        
        user_features = {}
        defaults = {
            'age': 30, 'bmi': 25, 'heart_rate': 72, 'systolic_bp': 120,
            'diastolic_bp': 80, 'cholesterol': 200, 'glucose': 100,
            'exercise_frequency': 2, 'sleep_hours': 7, 'stress_level': 5
        }
        
        for feature in features:
            user_features[feature] = user_data.get(feature, defaults[feature])
        
        # Add derived features
        user_features['pulse_pressure'] = (
            user_features['systolic_bp'] - user_features['diastolic_bp']
        )
        user_features['health_score'] = (
            5 - abs(user_features['bmi'] - 22) / 5 +
            (8 - user_features['stress_level']) +
            user_features['exercise_frequency'] +
            (user_features['sleep_hours'] - 4) / 2
        )
        user_features['health_score'] = max(0, min(20, user_features['health_score']))
        
        # Scale features
        feature_order = ['age', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                        'cholesterol', 'glucose', 'exercise_frequency', 'sleep_hours',
                        'stress_level', 'pulse_pressure', 'health_score']
        
        user_array = np.array([[user_features[f] for f in feature_order]])
        user_scaled = self.scaler.transform(user_array)
        
        # Predict cluster
        predicted_cluster = self.kmeans_model.predict(user_scaled)[0]
        
        # Get cluster information
        cluster_info = self.cluster_profiles.get(predicted_cluster, {})
        
        return {
            'cluster': predicted_cluster,
            'description': cluster_info.get('description', 'Unknown cluster'),
            'similar_users': cluster_info.get('size', 0),
            'characteristics': cluster_info.get('characteristics', []),
            'risk_level': cluster_info.get('risk_level', 'Unknown'),
            'percentage': cluster_info.get('percentage', 0)
        }
    
    def get_cluster_visualization(self):
        """Create cluster visualization using PCA"""
        if not self.is_fitted:
            return None
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'PC1': self.pca_features[:, 0],
            'PC2': self.pca_features[:, 1],
            'Cluster': self.health_data['kmeans_cluster'].astype(str),
            'Age': self.health_data['age'],
            'BMI': self.health_data['bmi'],
            'Health_Score': self.health_data['health_score']
        })
        
        # Create interactive plot
        fig = px.scatter(
            plot_data, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            size='Health_Score',
            hover_data=['Age', 'BMI'],
            title='Health Profile Clusters (PCA Visualization)',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
        )
        
        fig.update_layout(
            showlegend=True,
            width=700,
            height=500
        )
        
        return fig
    
    def get_feature_importance_plot(self):
        """Create feature importance visualization"""
        if not self.feature_importance:
            return None
        
        features = list(self.feature_importance.keys())
        importance = list(self.feature_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        features = [features[i] for i in sorted_indices]
        importance = [importance[i] for i in sorted_indices]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance in Health Clustering',
            labels={'x': 'Importance Score', 'y': 'Health Features'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            width=600,
            height=400
        )
        
        return fig
    
    def get_cluster_summary(self):
        """Get summary statistics for all clusters"""
        summary = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            summary[f"Cluster {cluster_id}"] = {
                'Description': profile['description'],
                'Size': f"{profile['size']} users ({profile['percentage']:.1f}%)",
                'Risk Level': profile['risk_level'],
                'Avg Age': f"{profile['avg_age']:.1f} years",
                'Avg BMI': f"{profile['avg_bmi']:.1f}",
                'Avg Health Score': f"{profile['avg_health_score']:.1f}/20"
            }
        
        return summary
    
    def get_recommendations_for_cluster(self, cluster_id):
        """Get health recommendations for a specific cluster"""
        recommendations = {
            0: {  # Healthy Young Adults
                'fitness': [
                    "Maintain current exercise routine",
                    "Try new activities to prevent boredom",
                    "Consider strength training 2-3x per week"
                ],
                'nutrition': [
                    "Focus on whole foods and balanced meals",
                    "Stay hydrated throughout the day",
                    "Consider meal prep for consistency"
                ],
                'lifestyle': [
                    "Maintain good sleep hygiene",
                    "Practice stress management techniques",
                    "Regular health checkups for prevention"
                ]
            },
            1: {  # Moderate Risk
                'fitness': [
                    "Gradually increase exercise frequency",
                    "Start with 30 minutes of walking daily",
                    "Add resistance training twice weekly"
                ],
                'nutrition': [
                    "Reduce processed food intake",
                    "Increase fiber and protein consumption",
                    "Monitor portion sizes"
                ],
                'lifestyle': [
                    "Improve sleep quality and duration",
                    "Manage stress through meditation or yoga",
                    "Regular monitoring of vital signs"
                ]
            },
            2: {  # High Risk
                'fitness': [
                    "Start with low-impact exercises",
                    "Consult doctor before starting new programs",
                    "Focus on consistency over intensity"
                ],
                'nutrition': [
                    "Follow DASH or Mediterranean diet",
                    "Limit sodium and sugar intake",
                    "Work with a nutritionist"
                ],
                'lifestyle': [
                    "Prioritize medication compliance",
                    "Regular medical monitoring",
                    "Stress reduction is critical"
                ]
            }
        }
        
        return recommendations.get(cluster_id, recommendations[1])  # Default to moderate risk

# Example usage
if __name__ == "__main__":
    # Initialize clustering model
    clustering = HealthClustering()
    
    # Example user data
    user_data = {
        'age': 35,
        'bmi': 26,
        'heart_rate': 75,
        'systolic_bp': 125,
        'diastolic_bp': 80,
        'cholesterol': 210,
        'glucose': 105,
        'exercise_frequency': 3,
        'sleep_hours': 7,
        'stress_level': 4
    }
    
    # Predict cluster
    result = clustering.predict_cluster(user_data)
    print(f"User belongs to: {result['description']}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Similar users: {result['similar_users']}")