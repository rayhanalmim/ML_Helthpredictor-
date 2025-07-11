import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional, Tuple

class DatabaseManager:
    def __init__(self, db_path='health_monitoring.db'):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database and create tables if they don't exist"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        
        # Create tables
        self._create_tables()
        
    def _create_tables(self):
        """Create all necessary tables"""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                bmi REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Health data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                heart_rate INTEGER,
                systolic_bp INTEGER,
                diastolic_bp INTEGER,
                glucose REAL,
                temperature REAL,
                cholesterol REAL,
                weight REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Health conditions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                condition_name TEXT,
                diagnosed_date DATE,
                severity TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Medications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                medication_name TEXT,
                dosage TEXT,
                frequency TEXT,
                start_date DATE,
                end_date DATE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Symptoms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symptoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                symptom_text TEXT,
                severity INTEGER,
                timestamp TIMESTAMP,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                category TEXT,
                recommendation_text TEXT,
                priority INTEGER,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Anomalies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                anomaly_type TEXT,
                severity TEXT,
                affected_vitals TEXT,
                description TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feedback_type TEXT,
                rating INTEGER,
                comment TEXT,
                feature_category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        print("Database initialized successfully")
    
    def create_user(self, user_data: Dict) -> str:
        """Create a new user"""
        cursor = self.connection.cursor()
        
        user_id = user_data.get('user_id', f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, name, age, gender, height, weight, bmi, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            user_data.get('name'),
            user_data.get('age'),
            user_data.get('gender'),
            user_data.get('height'),
            user_data.get('weight'),
            user_data.get('bmi'),
            datetime.now()
        ))
        
        self.connection.commit()
        return user_id
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def update_user(self, user_id: str, user_data: Dict) -> bool:
        """Update user information"""
        cursor = self.connection.cursor()
        
        # Build dynamic update query
        update_fields = []
        values = []
        
        for field in ['name', 'age', 'gender', 'height', 'weight', 'bmi']:
            if field in user_data:
                update_fields.append(f"{field} = ?")
                values.append(user_data[field])
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = ?")
        values.append(datetime.now())
        values.append(user_id)
        
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE user_id = ?"
        cursor.execute(query, values)
        
        self.connection.commit()
        return cursor.rowcount > 0
    
    def add_health_data(self, user_id: str, health_data: Dict) -> int:
        """Add health data entry"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO health_data 
            (user_id, timestamp, heart_rate, systolic_bp, diastolic_bp, 
             glucose, temperature, cholesterol, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            health_data.get('timestamp', datetime.now()),
            health_data.get('heart_rate'),
            health_data.get('systolic_bp'),
            health_data.get('diastolic_bp'),
            health_data.get('glucose'),
            health_data.get('temperature'),
            health_data.get('cholesterol'),
            health_data.get('weight')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_health_data(self, user_id: str, days: int = 30, limit: int = None) -> List[Dict]:
        """Get health data for user within specified days"""
        cursor = self.connection.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        query = '''
            SELECT * FROM health_data 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (user_id, start_date))
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_latest_vitals(self, user_id: str) -> Optional[Dict]:
        """Get latest vital signs for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT * FROM health_data 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (user_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def add_health_condition(self, user_id: str, condition_data: Dict) -> int:
        """Add health condition for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO health_conditions 
            (user_id, condition_name, diagnosed_date, severity, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            condition_data.get('condition_name'),
            condition_data.get('diagnosed_date'),
            condition_data.get('severity'),
            condition_data.get('notes')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_health_conditions(self, user_id: str) -> List[Dict]:
        """Get all health conditions for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT * FROM health_conditions 
            WHERE user_id = ? 
            ORDER BY diagnosed_date DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def add_medication(self, user_id: str, medication_data: Dict) -> int:
        """Add medication for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO medications 
            (user_id, medication_name, dosage, frequency, start_date, end_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            medication_data.get('medication_name'),
            medication_data.get('dosage'),
            medication_data.get('frequency'),
            medication_data.get('start_date'),
            medication_data.get('end_date'),
            medication_data.get('notes')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_active_medications(self, user_id: str) -> List[Dict]:
        """Get active medications for user"""
        cursor = self.connection.cursor()
        
        current_date = datetime.now().date()
        
        cursor.execute('''
            SELECT * FROM medications 
            WHERE user_id = ? 
            AND (end_date IS NULL OR end_date >= ?)
            ORDER BY start_date DESC
        ''', (user_id, current_date))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def add_symptom_analysis(self, user_id: str, symptom_data: Dict) -> int:
        """Add symptom analysis result"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO symptoms 
            (user_id, symptom_text, severity, timestamp, analysis_result)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            symptom_data.get('symptom_text'),
            symptom_data.get('severity'),
            symptom_data.get('timestamp', datetime.now()),
            json.dumps(symptom_data.get('analysis_result'))
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_symptom_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get symptom history for user"""
        cursor = self.connection.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM symptoms 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (user_id, start_date))
        
        rows = cursor.fetchall()
        symptoms = []
        
        for row in rows:
            symptom = dict(row)
            if symptom['analysis_result']:
                symptom['analysis_result'] = json.loads(symptom['analysis_result'])
            symptoms.append(symptom)
        
        return symptoms
    
    def add_recommendation(self, user_id: str, recommendation_data: Dict) -> int:
        """Add recommendation for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO recommendations 
            (user_id, category, recommendation_text, priority)
            VALUES (?, ?, ?, ?)
        ''', (
            user_id,
            recommendation_data.get('category'),
            recommendation_data.get('recommendation_text'),
            recommendation_data.get('priority', 1)
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_active_recommendations(self, user_id: str) -> List[Dict]:
        """Get active recommendations for user"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT * FROM recommendations 
            WHERE user_id = ? AND status = 'active'
            ORDER BY priority DESC, created_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def complete_recommendation(self, recommendation_id: int) -> bool:
        """Mark recommendation as completed"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            UPDATE recommendations 
            SET status = 'completed', completed_at = ?
            WHERE id = ?
        ''', (datetime.now(), recommendation_id))
        
        self.connection.commit()
        return cursor.rowcount > 0
    
    def add_anomaly(self, user_id: str, anomaly_data: Dict) -> int:
        """Add detected anomaly"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO anomalies 
            (user_id, timestamp, anomaly_type, severity, affected_vitals, description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            anomaly_data.get('timestamp', datetime.now()),
            anomaly_data.get('anomaly_type'),
            anomaly_data.get('severity'),
            json.dumps(anomaly_data.get('affected_vitals', [])),
            anomaly_data.get('description')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_recent_anomalies(self, user_id: str, days: int = 7) -> List[Dict]:
        """Get recent anomalies for user"""
        cursor = self.connection.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM anomalies 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (user_id, start_date))
        
        rows = cursor.fetchall()
        anomalies = []
        
        for row in rows:
            anomaly = dict(row)
            if anomaly['affected_vitals']:
                anomaly['affected_vitals'] = json.loads(anomaly['affected_vitals'])
            anomalies.append(anomaly)
        
        return anomalies
    
    def add_user_feedback(self, user_id: str, feedback_data: Dict) -> int:
        """Add user feedback"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback 
            (user_id, feedback_type, rating, comment, feature_category)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            feedback_data.get('feedback_type'),
            feedback_data.get('rating'),
            feedback_data.get('comment'),
            feedback_data.get('feature_category')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        cursor = self.connection.cursor()
        
        # Health data count
        cursor.execute('SELECT COUNT(*) as count FROM health_data WHERE user_id = ?', (user_id,))
        health_data_count = cursor.fetchone()['count']
        
        # Anomalies count
        cursor.execute('SELECT COUNT(*) as count FROM anomalies WHERE user_id = ?', (user_id,))
        anomalies_count = cursor.fetchone()['count']
        
        # Active recommendations count
        cursor.execute('''
            SELECT COUNT(*) as count FROM recommendations 
            WHERE user_id = ? AND status = 'active'
        ''', (user_id,))
        active_recommendations = cursor.fetchone()['count']
        
        # Health conditions count
        cursor.execute('SELECT COUNT(*) as count FROM health_conditions WHERE user_id = ?', (user_id,))
        conditions_count = cursor.fetchone()['count']
        
        # Latest vital signs
        latest_vitals = self.get_latest_vitals(user_id)
        
        return {
            'health_data_points': health_data_count,
            'total_anomalies': anomalies_count,
            'active_recommendations': active_recommendations,
            'health_conditions': conditions_count,
            'latest_vitals': latest_vitals,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_health_trends(self, user_id: str, days: int = 30) -> Dict:
        """Get health trends over specified period"""
        cursor = self.connection.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                AVG(heart_rate) as avg_heart_rate,
                AVG(systolic_bp) as avg_systolic_bp,
                AVG(diastolic_bp) as avg_diastolic_bp,
                AVG(glucose) as avg_glucose,
                AVG(temperature) as avg_temperature,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading,
                COUNT(*) as total_readings
            FROM health_data 
            WHERE user_id = ? AND timestamp >= ?
        ''', (user_id, start_date))
        
        trends = dict(cursor.fetchone())
        
        # Get daily averages for trend calculation
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                AVG(heart_rate) as daily_hr,
                AVG(systolic_bp) as daily_sys_bp,
                AVG(glucose) as daily_glucose
            FROM health_data 
            WHERE user_id = ? AND timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (user_id, start_date))
        
        daily_trends = [dict(row) for row in cursor.fetchall()]
        trends['daily_trends'] = daily_trends
        
        return trends
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data"""
        export_data = {
            'user_info': self.get_user(user_id),
            'health_data': self.get_health_data(user_id, days=365),
            'health_conditions': self.get_health_conditions(user_id),
            'medications': self.get_active_medications(user_id),
            'symptom_history': self.get_symptom_history(user_id, days=90),
            'recommendations': self.get_active_recommendations(user_id),
            'anomalies': self.get_recent_anomalies(user_id, days=90),
            'statistics': self.get_user_statistics(user_id),
            'trends': self.get_health_trends(user_id),
            'export_date': datetime.now().isoformat()
        }
        
        return export_data
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"health_monitoring_backup_{timestamp}.db"
        
        # Create backup
        backup_conn = sqlite3.connect(backup_path)
        self.connection.backup(backup_conn)
        backup_conn.close()
        
        return backup_path
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None

# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Create sample user
    user_data = {
        'name': 'John Doe',
        'age': 35,
        'gender': 'Male',
        'height': 175.0,
        'weight': 75.0,
        'bmi': 24.5
    }
    
    user_id = db_manager.create_user(user_data)
    print(f"Created user: {user_id}")
    
    # Add sample health data
    health_data = {
        'heart_rate': 72,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'glucose': 95,
        'temperature': 98.6
    }
    
    health_id = db_manager.add_health_data(user_id, health_data)
    print(f"Added health data with ID: {health_id}")
    
    # Get user statistics
    stats = db_manager.get_user_statistics(user_id)
    print(f"User statistics: {stats}")
    
    # Close connection
    db_manager.close_connection()