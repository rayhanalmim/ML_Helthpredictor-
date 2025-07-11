import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class SymptomChecker:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.naive_bayes_model = MultinomialNB()
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize medical knowledge base
        self._create_medical_database()
        self._create_symptom_database()
        self._train_models()
        
    def _create_medical_database(self):
        """Create a comprehensive medical knowledge base"""
        self.medical_conditions = {
            'Common Cold': {
                'symptoms': [
                    'runny nose', 'nasal congestion', 'sneezing', 'cough', 'sore throat',
                    'mild headache', 'low grade fever', 'fatigue', 'watery eyes',
                    'stuffy nose', 'throat irritation', 'mild body aches'
                ],
                'description': 'A viral infection of the upper respiratory tract',
                'urgency': 'low',
                'recommendations': [
                    'Get plenty of rest',
                    'Stay hydrated with fluids',
                    'Use a humidifier',
                    'Gargle with warm salt water',
                    'Consider over-the-counter medications for symptoms'
                ]
            },
            'Influenza (Flu)': {
                'symptoms': [
                    'high fever', 'severe headache', 'muscle aches', 'fatigue',
                    'dry cough', 'sore throat', 'runny nose', 'body aches',
                    'chills', 'weakness', 'loss of appetite', 'nausea'
                ],
                'description': 'A viral infection that attacks the respiratory system',
                'urgency': 'medium',
                'recommendations': [
                    'Rest and sleep as much as possible',
                    'Drink plenty of fluids',
                    'Consider antiviral medications if within 48 hours',
                    'Monitor fever and seek care if very high',
                    'Avoid contact with others to prevent spread'
                ]
            },
            'Migraine': {
                'symptoms': [
                    'severe headache', 'throbbing pain', 'nausea', 'vomiting',
                    'sensitivity to light', 'sensitivity to sound', 'visual disturbances',
                    'dizziness', 'fatigue', 'mood changes', 'neck stiffness'
                ],
                'description': 'A neurological condition causing severe headaches',
                'urgency': 'medium',
                'recommendations': [
                    'Rest in a dark, quiet room',
                    'Apply cold or warm compress',
                    'Stay hydrated',
                    'Avoid triggers like certain foods',
                    'Consider prescribed migraine medications'
                ]
            },
            'Hypertension': {
                'symptoms': [
                    'headache', 'dizziness', 'shortness of breath', 'chest pain',
                    'nosebleeds', 'fatigue', 'vision problems', 'irregular heartbeat',
                    'blood in urine', 'difficulty breathing'
                ],
                'description': 'High blood pressure that can lead to serious complications',
                'urgency': 'high',
                'recommendations': [
                    'Monitor blood pressure regularly',
                    'Reduce sodium intake',
                    'Exercise regularly',
                    'Maintain healthy weight',
                    'Take prescribed medications as directed'
                ]
            },
            'Diabetes': {
                'symptoms': [
                    'excessive thirst', 'frequent urination', 'extreme fatigue',
                    'blurred vision', 'slow healing wounds', 'weight loss',
                    'increased hunger', 'dry mouth', 'tingling in hands or feet'
                ],
                'description': 'A condition where blood sugar levels are too high',
                'urgency': 'high',
                'recommendations': [
                    'Monitor blood glucose levels',
                    'Follow a diabetes-friendly diet',
                    'Exercise regularly',
                    'Take medications as prescribed',
                    'Regular check-ups with healthcare provider'
                ]
            },
            'Heart Attack': {
                'symptoms': [
                    'chest pain', 'pressure in chest', 'shortness of breath',
                    'pain in arms', 'pain in jaw', 'nausea', 'sweating',
                    'dizziness', 'fatigue', 'indigestion-like pain'
                ],
                'description': 'A medical emergency where blood flow to heart is blocked',
                'urgency': 'critical',
                'recommendations': [
                    'Call emergency services immediately',
                    'Chew aspirin if not allergic',
                    'Stay calm and rest',
                    'Do not drive yourself to hospital',
                    'Have someone stay with you'
                ]
            },
            'Stroke': {
                'symptoms': [
                    'sudden weakness', 'face drooping', 'arm weakness',
                    'speech difficulty', 'sudden confusion', 'vision problems',
                    'severe headache', 'loss of coordination', 'dizziness'
                ],
                'description': 'A medical emergency where blood supply to brain is interrupted',
                'urgency': 'critical',
                'recommendations': [
                    'Call emergency services immediately',
                    'Note time symptoms started',
                    'Do not give food or water',
                    'Keep person calm and comfortable',
                    'Monitor breathing and consciousness'
                ]
            },
            'Anxiety': {
                'symptoms': [
                    'excessive worry', 'restlessness', 'fatigue', 'difficulty concentrating',
                    'irritability', 'muscle tension', 'sleep problems', 'rapid heartbeat',
                    'sweating', 'trembling', 'nausea', 'dizziness'
                ],
                'description': 'A mental health condition characterized by excessive worry',
                'urgency': 'medium',
                'recommendations': [
                    'Practice deep breathing exercises',
                    'Try relaxation techniques',
                    'Exercise regularly',
                    'Limit caffeine intake',
                    'Consider counseling or therapy'
                ]
            },
            'Depression': {
                'symptoms': [
                    'persistent sadness', 'loss of interest', 'fatigue',
                    'sleep problems', 'appetite changes', 'difficulty concentrating',
                    'feelings of worthlessness', 'hopelessness', 'irritability'
                ],
                'description': 'A mental health condition causing persistent feelings of sadness',
                'urgency': 'medium',
                'recommendations': [
                    'Seek professional mental health support',
                    'Maintain social connections',
                    'Exercise regularly',
                    'Maintain regular sleep schedule',
                    'Avoid alcohol and drugs'
                ]
            },
            'Gastroenteritis': {
                'symptoms': [
                    'nausea', 'vomiting', 'diarrhea', 'stomach cramps',
                    'fever', 'headache', 'muscle aches', 'dehydration',
                    'loss of appetite', 'fatigue'
                ],
                'description': 'Inflammation of stomach and intestines, often called stomach flu',
                'urgency': 'medium',
                'recommendations': [
                    'Stay hydrated with clear fluids',
                    'Rest and avoid solid foods initially',
                    'Gradually reintroduce bland foods',
                    'Avoid dairy and fatty foods',
                    'Seek care if severe dehydration occurs'
                ]
            }
        }
        
    def _create_symptom_database(self):
        """Create training data from medical knowledge base"""
        self.training_data = []
        self.training_labels = []
        
        for condition, info in self.medical_conditions.items():
            symptoms = info['symptoms']
            
            # Create multiple symptom combinations for each condition
            for i in range(len(symptoms)):
                for j in range(i+1, min(i+4, len(symptoms))):
                    # Create combinations of 2-3 symptoms
                    symptom_combination = ' '.join(symptoms[i:j+1])
                    self.training_data.append(symptom_combination)
                    self.training_labels.append(condition)
                    
                    # Add individual symptoms too
                    self.training_data.append(symptoms[i])
                    self.training_labels.append(condition)
        
        # Add some negative examples (symptoms not strongly associated with any condition)
        general_symptoms = [
            'tired', 'hungry', 'thirsty', 'sleepy', 'restless',
            'bored', 'excited', 'happy', 'calm', 'energetic'
        ]
        
        for symptom in general_symptoms:
            self.training_data.append(symptom)
            self.training_labels.append('General Symptoms')
    
    def _preprocess_text(self, text):
        """Preprocess text for NLP analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def _train_models(self):
        """Train the symptom classification models"""
        # Preprocess training data
        processed_training_data = [self._preprocess_text(text) for text in self.training_data]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(self.training_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_training_data, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train TF-IDF vectorizer
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Train Naive Bayes classifier
        self.naive_bayes_model.fit(X_train_tfidf, y_train)
        
        # Calculate accuracy
        accuracy = self.naive_bayes_model.score(X_test_tfidf, y_test)
        print(f"Symptom classifier accuracy: {accuracy:.3f}")
    
    def analyze_symptoms(self, symptom_text):
        """Analyze user symptoms and provide recommendations"""
        # Preprocess input text
        processed_text = self._preprocess_text(symptom_text)
        
        if not processed_text.strip():
            return {
                'possible_conditions': [],
                'recommendations': ['Please provide more specific symptoms for analysis'],
                'urgency': 'low'
            }
        
        # Get TF-IDF representation
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        
        # Get predictions from Naive Bayes
        nb_probabilities = self.naive_bayes_model.predict_proba(text_tfidf)[0]
        nb_classes = self.naive_bayes_model.classes_
        
        # Get similarity scores with known conditions
        similarity_scores = self._calculate_similarity_scores(processed_text)
        
        # Combine predictions
        final_predictions = self._combine_predictions(nb_probabilities, nb_classes, similarity_scores)
        
        # Generate response
        response = self._generate_response(final_predictions, symptom_text)
        
        return response
    
    def _calculate_similarity_scores(self, processed_text):
        """Calculate similarity scores with known conditions"""
        similarity_scores = {}
        
        for condition, info in self.medical_conditions.items():
            # Create text representation of condition symptoms
            condition_text = ' '.join(info['symptoms'])
            condition_processed = self._preprocess_text(condition_text)
            
            # Calculate TF-IDF similarity
            combined_texts = [processed_text, condition_processed]
            tfidf_matrix = TfidfVectorizer().fit_transform(combined_texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            similarity_scores[condition] = similarity
        
        return similarity_scores
    
    def _combine_predictions(self, nb_probabilities, nb_classes, similarity_scores):
        """Combine Naive Bayes predictions with similarity scores"""
        combined_scores = {}
        
        # Add Naive Bayes scores
        for i, class_idx in enumerate(nb_classes):
            condition = self.label_encoder.inverse_transform([class_idx])[0]
            if condition != 'General Symptoms':
                combined_scores[condition] = nb_probabilities[i] * 0.6  # Weight NB predictions
        
        # Add similarity scores
        for condition, similarity in similarity_scores.items():
            if condition in combined_scores:
                combined_scores[condition] += similarity * 0.4  # Weight similarity scores
            else:
                combined_scores[condition] = similarity * 0.4
        
        # Sort by combined score
        sorted_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:5]  # Top 5 predictions
    
    def _generate_response(self, predictions, original_text):
        """Generate response based on predictions"""
        if not predictions:
            return {
                'possible_conditions': [],
                'recommendations': ['Unable to analyze symptoms. Please consult a healthcare provider.'],
                'urgency': 'medium'
            }
        
        # Get top predictions
        possible_conditions = []
        max_urgency_level = 0
        urgency_mapping = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        for condition, confidence in predictions:
            if confidence > 0.1:  # Only include conditions with reasonable confidence
                condition_info = self.medical_conditions[condition]
                possible_conditions.append({
                    'name': condition,
                    'confidence': confidence,
                    'description': condition_info['description']
                })
                
                # Track highest urgency level
                urgency_level = urgency_mapping.get(condition_info['urgency'], 1)
                max_urgency_level = max(max_urgency_level, urgency_level)
        
        # Determine overall urgency
        urgency_reverse_mapping = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        overall_urgency = urgency_reverse_mapping[max_urgency_level]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(possible_conditions, overall_urgency, original_text)
        
        return {
            'possible_conditions': possible_conditions,
            'recommendations': recommendations,
            'urgency': overall_urgency
        }
    
    def _generate_recommendations(self, possible_conditions, urgency, original_text):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Add urgency-based recommendations
        if urgency == 'critical':
            recommendations.extend([
                "⚠️ URGENT: Seek immediate medical attention",
                "Call emergency services (911) right away",
                "Do not delay medical care"
            ])
        elif urgency == 'high':
            recommendations.extend([
                "Consult a healthcare provider today",
                "Monitor symptoms closely",
                "Seek immediate care if symptoms worsen"
            ])
        elif urgency == 'medium':
            recommendations.extend([
                "Consider scheduling an appointment with your doctor",
                "Monitor symptoms over the next 24-48 hours"
            ])
        else:
            recommendations.extend([
                "Monitor symptoms and consider self-care measures",
                "Consult a healthcare provider if symptoms persist or worsen"
            ])
        
        # Add condition-specific recommendations
        if possible_conditions:
            top_condition = possible_conditions[0]['name']
            condition_recs = self.medical_conditions[top_condition]['recommendations']
            recommendations.extend(condition_recs[:3])  # Add top 3 specific recommendations
        
        # Add general health recommendations
        recommendations.extend([
            "Stay hydrated and get adequate rest",
            "Maintain good hygiene practices",
            "Keep a symptom diary to track changes"
        ])
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def get_symptom_suggestions(self, partial_symptom):
        """Get symptom suggestions based on partial input"""
        suggestions = []
        partial_lower = partial_symptom.lower()
        
        for condition_info in self.medical_conditions.values():
            for symptom in condition_info['symptoms']:
                if partial_lower in symptom.lower() and symptom not in suggestions:
                    suggestions.append(symptom)
        
        return sorted(suggestions)[:10]  # Return top 10 suggestions
    
    def get_condition_info(self, condition_name):
        """Get detailed information about a specific condition"""
        if condition_name in self.medical_conditions:
            return self.medical_conditions[condition_name]
        return None
    
    def emergency_assessment(self, symptoms):
        """Quick assessment for emergency symptoms"""
        emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe headache',
            'confusion', 'seizure', 'loss of consciousness',
            'severe bleeding', 'choking', 'severe allergic reaction',
            'suicidal thoughts', 'severe abdominal pain'
        ]
        
        processed_symptoms = self._preprocess_text(symptoms.lower())
        
        for keyword in emergency_keywords:
            if keyword in processed_symptoms:
                return {
                    'emergency': True,
                    'message': f"⚠️ EMERGENCY: {keyword.title()} detected. Seek immediate medical attention!",
                    'action': "Call emergency services (911) immediately"
                }
        
        return {'emergency': False}
    
    def get_health_tips(self, condition=None):
        """Get general health tips or condition-specific tips"""
        if condition and condition in self.medical_conditions:
            return self.medical_conditions[condition]['recommendations']
        
        general_tips = [
            "Maintain a balanced diet with fruits and vegetables",
            "Exercise regularly for at least 30 minutes daily",
            "Get 7-9 hours of quality sleep each night",
            "Stay hydrated by drinking plenty of water",
            "Manage stress through relaxation techniques",
            "Avoid smoking and limit alcohol consumption",
            "Schedule regular health checkups",
            "Practice good hygiene habits"
        ]
        
        return general_tips
    
    def symptom_followup_questions(self, symptoms):
        """Generate follow-up questions to better understand symptoms"""
        questions = []
        
        if 'pain' in symptoms.lower():
            questions.extend([
                "On a scale of 1-10, how severe is the pain?",
                "Is the pain constant or does it come and go?",
                "What makes the pain better or worse?"
            ])
        
        if 'fever' in symptoms.lower():
            questions.extend([
                "What is your current temperature?",
                "How long have you had the fever?",
                "Are you experiencing chills or sweating?"
            ])
        
        if 'headache' in symptoms.lower():
            questions.extend([
                "Where exactly is the headache located?",
                "Is this different from your usual headaches?",
                "Are you experiencing any vision changes?"
            ])
        
        # Add general follow-up questions
        if not questions:
            questions.extend([
                "How long have you been experiencing these symptoms?",
                "Have you noticed any patterns or triggers?",
                "Are you taking any medications currently?"
            ])
        
        return questions[:5]  # Limit to 5 questions

# Example usage
if __name__ == "__main__":
    # Initialize symptom checker
    checker = SymptomChecker()
    
    # Example symptom analysis
    symptoms = "I have a severe headache, nausea, and sensitivity to light"
    
    result = checker.analyze_symptoms(symptoms)
    
    print("Symptom Analysis Results:")
    print(f"Urgency Level: {result['urgency']}")
    
    print("\nPossible Conditions:")
    for condition in result['possible_conditions']:
        print(f"- {condition['name']}: {condition['confidence']:.1%} confidence")
        print(f"  {condition['description']}")
    
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}")
    
    # Emergency assessment
    emergency_check = checker.emergency_assessment(symptoms)
    if emergency_check['emergency']:
        print(f"\n{emergency_check['message']}")
    
    # Follow-up questions
    questions = checker.symptom_followup_questions(symptoms)
    print("\nFollow-up Questions:")
    for q in questions:
        print(f"- {q}")