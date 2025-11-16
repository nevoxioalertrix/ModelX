"""
Dementia Risk Prediction - Binary Classification Model (Optimized)
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

print("="*80)
print("DEMENTIA RISK PREDICTION - OPTIMIZED PIPELINE")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('data.csv')
print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Sample if too large
if len(df) > 5000:
    print(f"   Sampling 5000 rows for faster processing...")
    df = df.sample(n=5000, random_state=RANDOM_STATE)

# Find target column
print("\n[2/6] Identifying target column...")
target_candidates = ['dementia', 'NACCALZD', 'NACCALZP', 'NACCUDSD', 'NACCETPR']
target_col = None
for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    # Find any binary column
    for col in df.columns:
        if 2 <= df[col].nunique() <= 5:
            target_col = col
            break

if not target_col:
    raise ValueError("No suitable target column found")

print(f"   Using: {target_col}")

# Prepare features
print("\n[3/6] Preparing features...")
exclude = [target_col, 'NACCID', 'PACKET', 'FORMVER', 'NACCADC']
feature_cols = [c for c in df.columns if c not in exclude]

# Keep only numeric features to avoid encoding issues
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"   Using {len(numeric_cols)} numeric features")

# Limit features if too many
if len(numeric_cols) > 50:
    numeric_cols = numeric_cols[:50]
    print(f"   Limited to top 50 features")

X = df[numeric_cols].copy()
y = df[target_col].copy()

# Convert target to binary
if y.nunique() > 2:
    threshold = y.median()
    y = (y > threshold).astype(int)

print(f"   Target distribution: {y.value_counts().to_dict()}")

# Handle missing values
print("\n[4/6] Handling missing values...")
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
print("\n[5/6] Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (fast settings)
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"   Model Accuracy: {accuracy:.4f}")

# Save model
print("\n[6/6] Saving model...")
model_path = Path('dementia_prediction_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'features': numeric_cols,
        'target': target_col,
        'accuracy': accuracy
    }, f)

print(f"   Saved: {model_path}")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model saved to: {model_path}")
print("="*80)