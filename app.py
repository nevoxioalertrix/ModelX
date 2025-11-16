from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import pickle
import json

app = Flask(__name__)

# Global variables to store model
model_data = {
    'model': None,
    'scaler': None,
    'feature_names': None,
    'is_trained': False,
    'accuracy': 0,
    'report': None
}

# Helper function to convert numpy/pandas types to Python native types
def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    # Check for numpy integer types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Check for numpy float types (NumPy 2.0 compatible)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    # Check for numpy bool
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Check for pandas Series
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    # Check for pandas DataFrame
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    # Recursively handle dictionaries
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    # Recursively handle lists
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    # Handle pandas Timestamp
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

class NACCPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load NACC dataset"""
        df = pd.read_csv(filepath, low_memory=False)
        return df
    
    def create_target(self, df):
        """Create target variable for prediction"""
        if 'DEMENTED' in df.columns:
            target = df['DEMENTED'].copy()
        elif 'CDRGLOB' in df.columns:
            target = (df['CDRGLOB'] >= 1).astype(int)
        else:
            raise ValueError("No suitable target variable found")
        return target
    
    def select_features(self, df):
        """Select relevant features that exist in the dataset"""
        feature_groups = {
            'Demographics': ['SEX', 'EDUC', 'MARISTAT', 'HANDED'],
            'Cognitive Tests': ['NACCMMSE', 'MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'CDRSUM'],
            'Medical History': ['CVHATT', 'CBSTROKE', 'DIABETES', 'HYPERTEN', 'HYPERCHO'],
            'Neuropsych Symptoms': ['DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'APA'],
            'Functional Activities': ['BILLS', 'TAXES', 'SHOPPING', 'STOVE', 'MEALPREP'],
            'Physical Exam': ['BPSYS', 'BPDIAS', 'HRATE', 'WEIGHT', 'HEIGHT']
        }
        
        available_features = []
        for group_name, features in feature_groups.items():
            available = [f for f in features if f in df.columns]
            available_features.extend(available)
        
        self.feature_names = available_features
        return available_features
    
    def preprocess_data(self, df, features, target):
        """Clean and prepare data"""
        X = df[features].copy()
        y = target.copy()
        
        # Remove rows where target is missing
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Handle missing values in features
        for col in X.columns:
            if X[col].isna().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else -1, inplace=True)
        
        # Convert all to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].median(), inplace=True)
        
        return X, y
    
    def train_model(self, X, y):
        """Train the prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Demented'], output_dict=True)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return accuracy, cm, report, importance_df, X_test, y_test, y_pred
    
    def predict(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        X_new = new_data[self.feature_names].copy()
        
        # Handle missing values
        for col in X_new.columns:
            if X_new[col].isna().any():
                X_new[col].fillna(X_new[col].median(), inplace=True)
        
        # Convert to numeric
        for col in X_new.columns:
            X_new[col] = pd.to_numeric(X_new[col], errors='coerce')
            X_new[col].fillna(0, inplace=True)
        
        # Scale
        X_scaled = self.scaler.transform(X_new)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = pd.DataFrame({
            'Prediction': ['Normal' if p == 0 else 'Demented' for p in predictions],
            'Confidence': [float(max(prob) * 100) for prob in probabilities],
            'Probability_Normal': [float(prob[0] * 100) for prob in probabilities],
            'Probability_Demented': [float(prob[1] * 100) for prob in probabilities]
        })
        
        return results

# Routes
@app.route('/')
def home():
    return render_template('index.html', is_trained=model_data['is_trained'])

@app.route('/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = os.path.join('uploads', 'train_data.csv')
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Train model
        predictor = NACCPredictor()
        df = predictor.load_data(filepath)
        target = predictor.create_target(df)
        features = predictor.select_features(df)
        X, y = predictor.preprocess_data(df, features, target)
        accuracy, cm, report, importance_df, X_test, y_test, y_pred = predictor.train_model(X, y)
        
        # Store model
        model_data['model'] = predictor.model
        model_data['scaler'] = predictor.scaler
        model_data['feature_names'] = predictor.feature_names
        model_data['is_trained'] = True
        model_data['accuracy'] = float(accuracy)
        model_data['report'] = report
        
        # Save model to file
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Demented'],
                    yticklabels=['Normal', 'Demented'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        
        img2 = io.BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        img2.seek(0)
        importance_plot_url = base64.b64encode(img2.getvalue()).decode()
        plt.close()
        
        # Convert all data to JSON-serializable format
        response_data = {
            'success': True,
            'accuracy': f'{float(accuracy) * 100:.2f}%',
            'samples': int(len(X)),
            'features': int(len(features)),
            'confusion_matrix': convert_to_serializable(cm),
            'confusion_matrix_plot': plot_url,
            'feature_importance_plot': importance_plot_url,
            'report': {
                'normal': {
                    'precision': f"{float(report['Normal']['precision']):.2f}",
                    'recall': f"{float(report['Normal']['recall']):.2f}",
                    'f1-score': f"{float(report['Normal']['f1-score']):.2f}"
                },
                'demented': {
                    'precision': f"{float(report['Demented']['precision']):.2f}",
                    'recall': f"{float(report['Demented']['recall']):.2f}",
                    'f1-score': f"{float(report['Demented']['f1-score']):.2f}"
                }
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_data['is_trained']:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = os.path.join('uploads', 'predict_data.csv')
        file.save(filepath)
        
        # Load data
        df = pd.read_csv(filepath, low_memory=False)
        
        # Make predictions
        predictor = NACCPredictor()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        
        results = predictor.predict(df)
        
        # Save results
        output_path = os.path.join('uploads', 'predictions.csv')
        results.to_csv(output_path, index=False)
        
        # Convert results to JSON-serializable format
        predictions_list = convert_to_serializable(results.to_dict('records'))
        
        normal_count = int((results['Prediction'] == 'Normal').sum())
        demented_count = int((results['Prediction'] == 'Demented').sum())
        
        response_data = {
            'success': True,
            'predictions': predictions_list,
            'total_samples': int(len(results)),
            'normal_count': normal_count,
            'demented_count': demented_count,
            'download_url': '/download/predictions'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/predictions')
def download_predictions():
    try:
        return send_file('uploads/predictions.csv', 
                        as_attachment=True,
                        download_name='predictions.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info')
def model_info():
    if model_data['is_trained']:
        return jsonify({
            'is_trained': True,
            'accuracy': f"{float(model_data['accuracy']) * 100:.2f}%",
            'features_count': int(len(model_data['feature_names'])),
            'features': model_data['feature_names']
        })
    else:
        return jsonify({'is_trained': False})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)