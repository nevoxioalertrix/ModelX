import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class NACCPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load NACC dataset"""
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Rows: {df.shape[0]:,}")
        print(f"  - Columns: {df.shape[1]:,}")
        return df
    
    def create_target(self, df):
        """Create target variable for prediction"""
        print("\n" + "=" * 60)
        print("STEP 2: Creating Target Variable")
        print("=" * 60)
        
        # Try multiple columns to create dementia status
        if 'DEMENTED' in df.columns:
            target = df['DEMENTED'].copy()
            print("✓ Using DEMENTED column as target")
        elif 'CDRGLOB' in df.columns:
            # CDR Global: 0=normal, 0.5=questionable, 1+=dementia
            target = (df['CDRGLOB'] >= 1).astype(int)
            print("✓ Created target from CDRGLOB (CDR >= 1 = Demented)")
        elif 'NORMCOG' in df.columns and 'DEMENTED' in df.columns:
            # Use diagnosis variables
            target = df['DEMENTED'].copy()
            print("✓ Using DEMENTED from diagnosis")
        else:
            print("✗ No suitable target variable found")
            print("Available columns:", df.columns.tolist()[:20], "...")
            raise ValueError("Cannot create target variable")
        
        print(f"\nTarget Distribution:")
        print(f"  - Normal (0): {(target == 0).sum():,} ({(target == 0).sum()/len(target)*100:.1f}%)")
        print(f"  - Demented (1): {(target == 1).sum():,} ({(target == 1).sum()/len(target)*100:.1f}%)")
        
        return target
    
    def select_features(self, df):
        """Select relevant features that exist in the dataset"""
        print("\n" + "=" * 60)
        print("STEP 3: Selecting Features")
        print("=" * 60)
        
        # Define feature groups
        feature_groups = {
            'Demographics': ['SEX', 'EDUC', 'MARISTAT', 'HANDED'],
            'Cognitive Tests': ['NACCMMSE', 'MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'CDRSUM'],
            'Medical History': ['CVHATT', 'CBSTROKE', 'DIABETES', 'HYPERTEN', 'HYPERCHO'],
            'Neuropsych Symptoms': ['DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'APA'],
            'Functional Activities': ['BILLS', 'TAXES', 'SHOPPING', 'STOVE', 'MEALPREP'],
            'Physical Exam': ['BPSYS', 'BPDIAS', 'HRATE', 'WEIGHT', 'HEIGHT']
        }
        
        available_features = []
        
        print("\nFeature Selection Summary:")
        for group_name, features in feature_groups.items():
            available = [f for f in features if f in df.columns]
            available_features.extend(available)
            print(f"  {group_name}: {len(available)}/{len(features)} available")
        
        if len(available_features) == 0:
            raise ValueError("No features available! Check your CSV column names.")
        
        print(f"\n✓ Total features selected: {len(available_features)}")
        self.feature_names = available_features
        return available_features
    
    def preprocess_data(self, df, features, target):
        """Clean and prepare data"""
        print("\n" + "=" * 60)
        print("STEP 4: Preprocessing Data")
        print("=" * 60)
        
        X = df[features].copy()
        y = target.copy()
        
        # Remove rows where target is missing
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"✓ Removed {(~valid_mask).sum()} rows with missing target")
        
        # Handle missing values in features
        print("\nHandling missing values...")
        for col in X.columns:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else -1, inplace=True)
        
        print(f"✓ Handled missing values in all features")
        
        # Convert all to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].median(), inplace=True)
        
        print(f"\n✓ Final dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
        
        return X, y
    
    def train_model(self, X, y):
        """Train the prediction model"""
        print("\n" + "=" * 60)
        print("STEP 5: Training Model")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest...")
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
        print("✓ Model trained successfully")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "=" * 60)
        print("STEP 6: Model Evaluation")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Negatives:  {cm[0,0]:,}")
        print(f"  False Positives: {cm[0,1]:,}")
        print(f"  False Negatives: {cm[1,0]:,}")
        print(f"  True Positives:  {cm[1,1]:,}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Demented']))
        
        # Feature importance
        self.show_feature_importance(X.columns)
        
        return X_test, y_test, y_pred
    
    def show_feature_importance(self, feature_names):
        """Display top features"""
        print("\n" + "=" * 60)
        print("Top 10 Most Important Features")
        print("=" * 60)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']:20s} : {row['Importance']:.4f}")
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved: feature_importance.png")
        plt.close()
    
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
        
        # Scale
        X_scaled = self.scaler.transform(X_new)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = pd.DataFrame({
            'Prediction': ['Normal' if p == 0 else 'Demented' for p in predictions],
            'Confidence': [max(prob) for prob in probabilities],
            'Probability_Normal': probabilities[:, 0],
            'Probability_Demented': probabilities[:, 1]
        })
        
        return results

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("NACC DEMENTIA PREDICTION MODEL")
    print("=" * 60)
    
    # Initialize predictor
    predictor = NACCPredictor()
    
    # Step 1: Load your data file
    # CHANGE THIS PATH TO YOUR CSV FILE!
    csv_file = r"C:\Users\humai\Desktop\modelx\nacc_data.csv"
    
    try:
        df = predictor.load_data(csv_file)
        
        # Step 2: Create target
        target = predictor.create_target(df)
        
        # Step 3: Select features
        features = predictor.select_features(df)
        
        # Step 4: Preprocess
        X, y = predictor.preprocess_data(df, features, target)
        
        # Step 5 & 6: Train and evaluate
        X_test, y_test, y_pred = predictor.train_model(X, y)
        
        # Example predictions
        print("\n" + "=" * 60)
        print("Sample Predictions (First 5 Test Cases)")
        print("=" * 60)
        sample_predictions = predictor.predict(X_test.head(5))
        print(sample_predictions.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✓ MODEL TRAINING COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check 'feature_importance.png' for feature analysis")
        print("2. Use predictor.predict(new_data) for new predictions")
        
    except FileNotFoundError:
        print("\n❌ ERROR: CSV file not found!")
        print(f"Please update the csv_file path in the code.")
        print(f"Current path: {csv_file}")
        print(f"\nYour CSV file should be in: C:\\Users\\humai\\Desktop\\modelx\\")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()