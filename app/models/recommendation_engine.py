import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random
from collections import defaultdict, deque
import json

class HealthRecommendationEngine:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995
        
        # Knowledge base for recommendations
        self.fitness_recommendations = {
            'low_fitness': [
                "Start with 10-15 minutes of walking daily",
                "Try gentle stretching exercises",
                "Consider chair exercises if mobility is limited",
                "Gradually increase activity duration",
                "Focus on consistency over intensity"
            ],
            'moderate_fitness': [
                "Aim for 30 minutes of moderate exercise 5 days a week",
                "Include strength training 2-3 times per week",
                "Try activities like swimming, cycling, or brisk walking",
                "Add flexibility and balance exercises",
                "Consider group fitness classes"
            ],
            'high_fitness': [
                "Continue current exercise routine with variations",
                "Challenge yourself with high-intensity interval training",
                "Focus on sport-specific training if interested",
                "Include recovery days for muscle repair",
                "Consider training for fitness events or competitions"
            ]
        }
        
        self.nutrition_recommendations = {
            'weight_loss': [
                "Create a moderate caloric deficit (300-500 calories)",
                "Focus on whole foods: fruits, vegetables, lean proteins",
                "Control portion sizes using smaller plates",
                "Drink water before meals",
                "Limit processed foods and sugary drinks"
            ],
            'weight_maintenance': [
                "Maintain balanced macronutrient ratios",
                "Eat regular, well-spaced meals",
                "Include variety in your diet",
                "Stay hydrated throughout the day",
                "Practice mindful eating"
            ],
            'muscle_gain': [
                "Increase protein intake to 1.6-2.2g per kg body weight",
                "Eat in a slight caloric surplus",
                "Include post-workout protein and carbohydrates",
                "Focus on nutrient-dense whole foods",
                "Consider meal timing around workouts"
            ],
            'diabetes_management': [
                "Monitor carbohydrate intake and timing",
                "Choose complex carbohydrates over simple sugars",
                "Include fiber-rich foods in meals",
                "Eat regular, consistent meals",
                "Work with a dietitian for personalized meal planning"
            ],
            'heart_health': [
                "Follow a Mediterranean-style diet",
                "Limit sodium intake to less than 2300mg daily",
                "Include omega-3 rich foods like fish",
                "Increase soluble fiber intake",
                "Limit saturated and trans fats"
            ]
        }
        
        self.lifestyle_recommendations = {
            'stress_management': [
                "Practice deep breathing exercises daily",
                "Try meditation or mindfulness techniques",
                "Maintain a regular sleep schedule",
                "Engage in hobbies you enjoy",
                "Consider talking to a counselor if stress is overwhelming"
            ],
            'sleep_improvement': [
                "Maintain consistent sleep and wake times",
                "Create a relaxing bedtime routine",
                "Keep bedroom cool, dark, and quiet",
                "Avoid screens 1 hour before bedtime",
                "Limit caffeine intake after 2 PM"
            ],
            'preventive_care': [
                "Schedule regular health checkups",
                "Stay up to date with vaccinations",
                "Know your family health history",
                "Monitor key health metrics regularly",
                "Practice good hygiene habits"
            ]
        }
        
        # User interaction history for learning
        self.user_feedback_history = defaultdict(list)
        self.recommendation_effectiveness = defaultdict(float)
        
    def _calculate_health_state(self, user_data, health_data):
        """Calculate current health state for Q-learning"""
        # Extract key metrics
        age = user_data.get('age', 30)
        bmi = user_data.get('bmi', 25)
        
        if health_data:
            recent_vitals = health_data[-1]
            avg_heart_rate = recent_vitals.get('heart_rate', 75)
            avg_bp = recent_vitals.get('systolic_bp', 120)
            avg_glucose = recent_vitals.get('glucose', 100)
        else:
            avg_heart_rate = 75
            avg_bp = 120
            avg_glucose = 100
        
        # Discretize state space
        age_bucket = 'young' if age < 35 else 'middle' if age < 55 else 'senior'
        bmi_bucket = 'underweight' if bmi < 18.5 else 'normal' if bmi < 25 else 'overweight' if bmi < 30 else 'obese'
        hr_bucket = 'low' if avg_heart_rate < 60 else 'normal' if avg_heart_rate < 100 else 'high'
        bp_bucket = 'low' if avg_bp < 90 else 'normal' if avg_bp < 140 else 'high'
        glucose_bucket = 'low' if avg_glucose < 70 else 'normal' if avg_glucose < 140 else 'high'
        
        return f"{age_bucket}_{bmi_bucket}_{hr_bucket}_{bp_bucket}_{glucose_bucket}"
    
    def _get_possible_actions(self):
        """Get all possible recommendation actions"""
        actions = []
        
        # Fitness actions
        for fitness_level in self.fitness_recommendations:
            actions.extend([f"fitness_{fitness_level}_{i}" for i in range(len(self.fitness_recommendations[fitness_level]))])
        
        # Nutrition actions
        for nutrition_type in self.nutrition_recommendations:
            actions.extend([f"nutrition_{nutrition_type}_{i}" for i in range(len(self.nutrition_recommendations[nutrition_type]))])
        
        # Lifestyle actions
        for lifestyle_type in self.lifestyle_recommendations:
            actions.extend([f"lifestyle_{lifestyle_type}_{i}" for i in range(len(self.lifestyle_recommendations[lifestyle_type]))])
        
        return actions
    
    def _select_action(self, state):
        """Select action using epsilon-greedy strategy"""
        possible_actions = self._get_possible_actions()
        
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.choice(possible_actions)
        else:
            # Exploit: best known action
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return random.choice(possible_actions)
    
    def _calculate_reward(self, action, user_feedback, health_improvement):
        """Calculate reward based on user feedback and health improvement"""
        base_reward = 0
        
        # User feedback component
        if user_feedback == 'positive':
            base_reward += 10
        elif user_feedback == 'neutral':
            base_reward += 5
        elif user_feedback == 'negative':
            base_reward -= 5
        
        # Health improvement component
        if health_improvement > 0.1:
            base_reward += 15
        elif health_improvement > 0:
            base_reward += 5
        elif health_improvement < -0.1:
            base_reward -= 10
        
        return base_reward
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        current_q = self.q_table[state][action]
        
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.1, self.exploration_rate)
    
    def generate_recommendations(self, user_data, health_data, num_recommendations=3):
        """Generate personalized health recommendations"""
        state = self._calculate_health_state(user_data, health_data)
        
        # Get base recommendations
        fitness_recs = self._get_fitness_recommendations(user_data, health_data)
        nutrition_recs = self._get_nutrition_recommendations(user_data, health_data)
        lifestyle_recs = self._get_lifestyle_recommendations(user_data, health_data)
        
        # Apply Q-learning for personalization
        if state in self.q_table and self.q_table[state]:
            # Sort actions by Q-value
            sorted_actions = sorted(self.q_table[state].items(), key=lambda x: x[1], reverse=True)
            
            # Enhance recommendations based on learned preferences
            for action, q_value in sorted_actions[:5]:  # Top 5 learned actions
                if q_value > 5:  # Only consider well-performing actions
                    rec_text = self._action_to_recommendation(action)
                    if rec_text:
                        if action.startswith('fitness'):
                            fitness_recs.insert(0, rec_text)
                        elif action.startswith('nutrition'):
                            nutrition_recs.insert(0, rec_text)
                        elif action.startswith('lifestyle'):
                            lifestyle_recs.insert(0, rec_text)
        
        # Remove duplicates and limit recommendations
        fitness_recs = list(dict.fromkeys(fitness_recs))[:num_recommendations]
        nutrition_recs = list(dict.fromkeys(nutrition_recs))[:num_recommendations]
        lifestyle_recs = list(dict.fromkeys(lifestyle_recs))[:num_recommendations]
        
        return {
            'fitness': fitness_recs,
            'nutrition': nutrition_recs,
            'lifestyle': lifestyle_recs,
            'state': state
        }
    
    def _action_to_recommendation(self, action):
        """Convert action string back to recommendation text"""
        try:
            parts = action.split('_')
            category = parts[0]
            subcategory = '_'.join(parts[1:-1])
            index = int(parts[-1])
            
            if category == 'fitness' and subcategory in self.fitness_recommendations:
                return self.fitness_recommendations[subcategory][index]
            elif category == 'nutrition' and subcategory in self.nutrition_recommendations:
                return self.nutrition_recommendations[subcategory][index]
            elif category == 'lifestyle' and subcategory in self.lifestyle_recommendations:
                return self.lifestyle_recommendations[subcategory][index]
        except:
            pass
        return None
    
    def _get_fitness_recommendations(self, user_data, health_data):
        """Get fitness recommendations based on user profile"""
        recommendations = []
        
        age = user_data.get('age', 30)
        bmi = user_data.get('bmi', 25)
        
        # Determine fitness level based on profile
        if bmi > 30 or age > 65:
            fitness_level = 'low_fitness'
        elif bmi > 25 or age > 50:
            fitness_level = 'moderate_fitness'
        else:
            fitness_level = 'high_fitness'
        
        # Add specific recommendations based on health data
        if health_data:
            recent_vitals = health_data[-1]
            heart_rate = recent_vitals.get('heart_rate', 75)
            bp = recent_vitals.get('systolic_bp', 120)
            
            if heart_rate > 90:
                recommendations.append("Focus on stress-reducing exercises like yoga or tai chi")
            if bp > 140:
                recommendations.append("Include low-impact cardio like walking or swimming")
        
        # Add base recommendations
        recommendations.extend(self.fitness_recommendations[fitness_level][:3])
        
        return recommendations
    
    def _get_nutrition_recommendations(self, user_data, health_data):
        """Get nutrition recommendations based on user profile and health data"""
        recommendations = []
        
        bmi = user_data.get('bmi', 25)
        age = user_data.get('age', 30)
        
        # Determine primary nutrition goal
        if bmi > 25:
            nutrition_type = 'weight_loss'
        elif bmi < 18.5:
            nutrition_type = 'muscle_gain'
        else:
            nutrition_type = 'weight_maintenance'
        
        # Check for specific health conditions
        if health_data:
            recent_vitals = health_data[-1]
            glucose = recent_vitals.get('glucose', 100)
            bp = recent_vitals.get('systolic_bp', 120)
            
            if glucose > 126:
                nutrition_type = 'diabetes_management'
            elif bp > 140:
                nutrition_type = 'heart_health'
        
        # Add age-specific recommendations
        if age > 60:
            recommendations.append("Ensure adequate calcium and vitamin D intake")
            recommendations.append("Focus on protein to maintain muscle mass")
        
        # Add base recommendations
        recommendations.extend(self.nutrition_recommendations[nutrition_type][:3])
        
        return recommendations
    
    def _get_lifestyle_recommendations(self, user_data, health_data):
        """Get lifestyle recommendations"""
        recommendations = []
        
        age = user_data.get('age', 30)
        
        # Always include preventive care
        recommendations.extend(self.lifestyle_recommendations['preventive_care'][:2])
        
        # Add stress management if needed
        if health_data:
            recent_vitals = health_data[-1]
            heart_rate = recent_vitals.get('heart_rate', 75)
            
            if heart_rate > 85:
                recommendations.extend(self.lifestyle_recommendations['stress_management'][:2])
        
        # Add sleep recommendations
        recommendations.extend(self.lifestyle_recommendations['sleep_improvement'][:2])
        
        return recommendations
    
    def record_user_feedback(self, user_id, recommendations, feedback, health_improvement=0):
        """Record user feedback for learning"""
        self.user_feedback_history[user_id].append({
            'recommendations': recommendations,
            'feedback': feedback,
            'health_improvement': health_improvement,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update Q-table based on feedback
        state = recommendations.get('state', 'unknown')
        
        for category in ['fitness', 'nutrition', 'lifestyle']:
            for i, rec in enumerate(recommendations.get(category, [])):
                action = f"{category}_general_{i}"
                reward = self._calculate_reward(action, feedback, health_improvement)
                
                # Simulate next state (simplified)
                next_state = state  # In practice, this would be the state after following recommendations
                
                self.update_q_table(state, action, reward, next_state)
    
    def get_adaptive_recommendations(self, user_id, user_data, health_data):
        """Get recommendations that adapt based on user's history"""
        base_recommendations = self.generate_recommendations(user_data, health_data)
        
        # Check user history for preferences
        if user_id in self.user_feedback_history:
            history = self.user_feedback_history[user_id]
            
            # Analyze which types of recommendations worked well
            category_scores = {'fitness': 0, 'nutrition': 0, 'lifestyle': 0}
            
            for entry in history[-10:]:  # Last 10 interactions
                feedback = entry['feedback']
                improvement = entry['health_improvement']
                
                score_change = 1 if feedback == 'positive' else -0.5 if feedback == 'negative' else 0
                score_change += improvement * 2
                
                for category in category_scores:
                    if entry['recommendations'].get(category):
                        category_scores[category] += score_change
            
            # Adjust recommendations based on user preferences
            for category in category_scores:
                if category_scores[category] > 2:
                    # User responds well to this category, add more
                    additional_recs = self._get_additional_recommendations(category, user_data, health_data)
                    base_recommendations[category].extend(additional_recs[:2])
                elif category_scores[category] < -2:
                    # User doesn't respond well, reduce recommendations
                    base_recommendations[category] = base_recommendations[category][:2]
        
        return base_recommendations
    
    def _get_additional_recommendations(self, category, user_data, health_data):
        """Get additional recommendations for a specific category"""
        if category == 'fitness':
            return ["Try a new type of exercise to prevent boredom", "Consider working with a personal trainer"]
        elif category == 'nutrition':
            return ["Consider meal planning and prep", "Try tracking your food intake for better awareness"]
        elif category == 'lifestyle':
            return ["Set up a regular routine for health monitoring", "Consider joining a support group or health community"]
        return []
    
    def get_goal_based_recommendations(self, user_goals, user_data, health_data):
        """Generate recommendations based on specific user goals"""
        recommendations = {'fitness': [], 'nutrition': [], 'lifestyle': []}
        
        for goal in user_goals:
            if goal == 'weight_loss':
                recommendations['fitness'].extend([
                    "Combine cardio and strength training for optimal fat loss",
                    "Aim for 150 minutes of moderate cardio per week"
                ])
                recommendations['nutrition'].extend(self.nutrition_recommendations['weight_loss'][:2])
                
            elif goal == 'muscle_gain':
                recommendations['fitness'].extend([
                    "Focus on progressive overload in strength training",
                    "Allow adequate rest between training sessions"
                ])
                recommendations['nutrition'].extend(self.nutrition_recommendations['muscle_gain'][:2])
                
            elif goal == 'improved_cardiovascular_health':
                recommendations['fitness'].extend([
                    "Include regular aerobic exercise",
                    "Monitor heart rate during exercise"
                ])
                recommendations['nutrition'].extend(self.nutrition_recommendations['heart_health'][:2])
                
            elif goal == 'better_sleep':
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['sleep_improvement'][:3])
                
            elif goal == 'stress_reduction':
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['stress_management'][:3])
        
        # Remove duplicates
        for category in recommendations:
            recommendations[category] = list(dict.fromkeys(recommendations[category]))
        
        return recommendations
    
    def get_progress_tracking_recommendations(self, user_data, health_data):
        """Provide recommendations for tracking progress"""
        tracking_recs = [
            "Log your daily activities and meals",
            "Take progress photos monthly",
            "Track key health metrics weekly",
            "Set SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)",
            "Review and adjust your plan monthly"
        ]
        
        # Personalize based on user data
        if user_data.get('bmi', 25) > 25:
            tracking_recs.insert(0, "Monitor your weight weekly at the same time of day")
        
        if health_data and health_data[-1].get('systolic_bp', 120) > 130:
            tracking_recs.insert(0, "Monitor blood pressure daily")
        
        return tracking_recs[:5]

# Example usage
if __name__ == "__main__":
    # Initialize recommendation engine
    engine = HealthRecommendationEngine()
    
    # Example user data
    user_data = {
        'age': 35,
        'bmi': 27,
        'gender': 'Male'
    }
    
    health_data = [{
        'heart_rate': 85,
        'systolic_bp': 135,
        'diastolic_bp': 88,
        'glucose': 110,
        'temperature': 98.6
    }]
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(user_data, health_data)
    
    print("Fitness Recommendations:")
    for rec in recommendations['fitness']:
        print(f"- {rec}")
    
    print("\nNutrition Recommendations:")
    for rec in recommendations['nutrition']:
        print(f"- {rec}")
    
    print("\nLifestyle Recommendations:")
    for rec in recommendations['lifestyle']:
        print(f"- {rec}")
    
    # Simulate user feedback
    engine.record_user_feedback('user_123', recommendations, 'positive', health_improvement=0.1)
    
    # Get adaptive recommendations
    adaptive_recs = engine.get_adaptive_recommendations('user_123', user_data, health_data)
    print(f"\nAdaptive recommendations generated based on user feedback")