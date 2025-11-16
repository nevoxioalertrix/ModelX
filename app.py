from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import os
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import sys
import time  # Add this to the imports at the very top

# Check dependencies
print("\n" + "="*60)
print("CHECKING DEPENDENCIES")
print("="*60)
print(f"Python version: {sys.version}")

try:
    print(f"✓ pandas {pd.__version__}")
except:
    print("❌ pandas not found")

try:
    print(f"✓ numpy {np.__version__}")
except:
    print("❌ numpy not found")

try:
    from sklearn import __version__ as sklearn_version
    print(f"✓ scikit-learn {sklearn_version}")
except:
    print("❌ scikit-learn not found")

try:
    print(f"✓ matplotlib {matplotlib.__version__}")
except:
    print("❌ matplotlib not found")

try:
    print(f"✓ seaborn {sns.__version__}")
except:
    print("❌ seaborn not found")

try:
    import xgboost as xgb
    HAS_XGB = True
    print(f"✓ xgboost {xgb.__version__}")
except ImportError:
    HAS_XGB = False
    print("⚠️ xgboost not available (optional)")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print(f"✓ imbalanced-learn available")
except ImportError:
    HAS_SMOTE = False
    print("⚠️ imbalanced-learn not available (optional)")

print("="*60 + "\n")

app = Flask(__name__)
CORS(app)

class NACCPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.encoded_feature_names = []
        self.feature_summary = {}
        self.target_column = 'NACCALZD'
        self.scaler = None
        self.selector = None
        self.debug_mode = True  # Add this line
    
    def select_features(self, df):
        """Select relevant features that exist in the dataset"""
        feature_groups = {
            'Demographics': [
                'BIRTHYR', 'SEX', 'HISPANIC', 'HISPOR', 'HISPORX',
                'RACE', 'RACEX', 'RACESEC', 'RACESECX', 'RACETER', 'RACETERX',
                'PRIMLANG', 'EDUC', 'MARISTAT', 'HANDED'
            ],
            'Living Situation': [
                'LIVSITUA', 'NACCLIVS', 'RESIDENC'
            ],
            'Functional Activities': [
                'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 
                'MEALPREP', 'EVENTS', 'TV', 'REMOTE', 'TRAVEL'
            ],
            'Co-participant Info': [
                'INSEX', 'INEDUC', 'INRELTO', 'INLIVWTH'
            ],
            'Cognitive Tests': [
                'NACCMMSE', 'MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 
                'CDRSUM', 'CDRGLOB', 'NACCMOCA'
            ],
            'Behavioral Symptoms': [
                'DEL', 'DELSEV', 'HALL', 'HALLSEV', 'AGIT', 'AGITSEV',
                'DEPD', 'DEPDSEV', 'ANX', 'ANXSEV', 'ELAT', 'ELATSEV',
                'APA', 'APASEV', 'DISN', 'DISNSEV', 'IRR', 'IRRSEV',
                'MOT', 'MOTSEV', 'NITE', 'NITESEV', 'APP', 'APPSEV'
            ],
            'Physical Measurements': [
                'BPSYS', 'BPDIAS', 'HRATE', 'WEIGHT', 'HEIGHT'
            ],
            'Medical History': [
                'CVHATT', 'CBSTROKE', 'DIABETES', 'HYPERTEN', 'HYPERCHO',
                'DEP2YRS', 'ANXIETY', 'SMOKYRS', 'PACKSPER'
            ]
        }
        
        available_features = []
        feature_summary = {}
        
        for group_name, features in feature_groups.items():
            available = [f for f in features if f in df.columns]
            available_features.extend(available)
            feature_summary[group_name] = {
                'available': len(available),
                'total': len(features),
                'features': available
            }
        
        self.feature_names = available_features
        self.feature_summary = feature_summary
        
        return available_features
    
    def prepare_data(self, df, features):
        """Prepare data with proper missing value handling"""
        X = df[features].copy()
        
        # Fill numeric columns with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X[col].isnull().any():
                median_value = X[col].median()
                if pd.isna(median_value):
                    X[col].fillna(0, inplace=True)
                else:
                    X[col].fillna(median_value, inplace=True)
        
        # Fill categorical columns with mode or 'Unknown'
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if X[col].isnull().any():
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X[col].fillna(mode_value[0], inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
        
        # Convert all to numeric (one-hot encode categoricals)
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Final check: replace any remaining NaN with 0
        X_encoded = X_encoded.fillna(0)
        
        return X_encoded
    
    def train(self, file_path):
        """Train the model - OPTIMIZED FOR SPEED"""
        try:
            total_start = time.time()
            
            print(f"\n{'='*60}")
            print("⚡ FAST TRAINING MODE")
            print(f"{'='*60}\n")
            
            # Load data
            print("📂 Loading data...")
            start = time.time()
            df = pd.read_csv(file_path, low_memory=False)
            print(f"✓ Loaded {len(df)} records in {time.time()-start:.2f}s")
            
            # Check for possible target columns
            possible_targets = ['NACCALZD', 'NACCUDSD', 'NACCDAGE', 'dx', 'CDRGLOB']
            available_target = None
            
            for target in possible_targets:
                if target in df.columns:
                    available_target = target
                    self.target_column = target
                    print(f"✓ Found target column: {target}")
                    break
            
            if available_target is None:
                raise ValueError(f"No target column found. Expected one of: {possible_targets}")
            
            # Select features
            start = time.time()
            available_features = self.select_features(df)
            print(f"✓ Feature selection: {len(available_features)} features in {time.time()-start:.2f}s")
            
            # Check target distribution
            target_counts = df[self.target_column].value_counts()
            print(f"\n📊 Target distribution:")
            for val, count in target_counts.items():
                print(f"  Class {val}: {count} samples ({count/len(df)*100:.1f}%)")
            
            if len(target_counts) < 2:
                raise ValueError(f"Need at least 2 classes. Found only: {len(target_counts)}")
            
            # Remove rows where target is missing
            df_clean = df.dropna(subset=[self.target_column])
            print(f"\n✓ Clean dataset: {len(df_clean)} rows")
            
            if len(df_clean) < 10:
                raise ValueError(f"Not enough data! Only {len(df_clean)} rows after cleaning.")
            
            # SPEED BOOST 1: Aggressive sampling for large datasets
            max_training_samples = 30000  # Reduced from 100k
            if len(df_clean) > max_training_samples:
                print(f"\n⚡ Sampling {max_training_samples} rows for faster training...")
                df_clean = df_clean.sample(n=max_training_samples, random_state=42)
            
            # Prepare features
            start = time.time()
            print("\n🔧 Preparing features...")
            X = self.prepare_data(df_clean, available_features)
            y = df_clean[self.target_column].astype(int)
            print(f"✓ Data preparation: {time.time()-start:.2f}s")
            
            self.encoded_feature_names = X.columns.tolist()
            
            print(f"\n{'='*60}")
            print("🎯 TRAINING CONFIGURATION")
            print(f"{'='*60}")
            print(f"Features: {X.shape[1]}")
            print(f"Samples: {X.shape[0]}")
            print(f"Classes: {len(np.unique(y))}")
            
            # Split data
            start = time.time()
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            print(f"✓ Train/test split: {time.time()-start:.2f}s")
            
            # Convert to numpy arrays
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
            y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            
            # SPEED BOOST 2: Skip feature scaling (minimal impact on tree models)
            print("\n⚡ Skipping feature scaling for speed...")
            X_train_scaled = X_train_np
            X_test_scaled = X_test_np
            
            # SPEED BOOST 3: Feature Selection - fewer features
            print("🎯 Selecting best features...")
            n_features_to_select = min(30, X_train.shape[1])  # Reduced from 50
            selector = SelectKBest(mutual_info_classif, k=n_features_to_select)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train_np)
            X_test_selected = selector.transform(X_test_scaled)
            print(f"✓ Selected top {n_features_to_select} features")
            
            # SPEED BOOST 4: Skip SMOTE (time-consuming)
            print("⚡ Skipping SMOTE for speed (using class_weight instead)...")
            X_train_balanced = X_train_selected
            y_train_balanced = y_train_np
            
            # SPEED BOOST 5: Simpler, faster model
            print(f"\n{'='*60}")
            print("🚀 TRAINING OPTIMIZED MODEL")
            print(f"{'='*60}")
            
            start = time.time()
            
            # Single optimized Random Forest (faster than ensemble)
            self.model = RandomForestClassifier(
                n_estimators=100,          # Reduced from 200
                max_depth=15,              # Reduced from 20
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,                 # Use all CPU cores
                class_weight='balanced',   # Handle imbalance
                bootstrap=True,
                warm_start=False,
                max_samples=0.8            # Speed boost: subsample
            )
            
            print("⏳ Training Random Forest...")
            self.model.fit(X_train_balanced, y_train_balanced)
            training_time = time.time() - start
            print(f"✓ Training completed: {training_time:.2f}s")
            
            # Store preprocessing
            self.scaler = None  # Not using scaler
            self.selector = selector
            
            # Evaluate
            print("\n📊 Evaluating...")
            y_pred = self.model.predict(X_test_selected)
            y_pred_proba = self.model.predict_proba(X_test_selected)
            accuracy = accuracy_score(y_test_np, y_pred)
            
            # SPEED BOOST 6: Fast cross-validation (3-fold instead of 5)
            print("🔄 Quick cross-validation (3-fold)...")
            start = time.time()
            
            cv_scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train_np)):
                print(f"  Fold {fold_idx + 1}/3...", end=' ')
                
                X_cv_train = X_train_selected[train_idx]
                X_cv_val = X_train_selected[val_idx]
                y_cv_train = y_train_np[train_idx]
                y_cv_val = y_train_np[val_idx]
                
                # Smaller model for CV
                temp_model = RandomForestClassifier(
                    n_estimators=50,       # Half the trees
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                temp_model.fit(X_cv_train, y_cv_train)
                score = temp_model.score(X_cv_val, y_cv_val)
                cv_scores.append(score)
                print(f"{score:.4f}")
            
            cv_accuracy = np.mean(cv_scores)
            print(f"✓ CV complete: {time.time()-start:.2f}s")
            
            print(f"\n{'='*60}")
            print("🎉 TRAINING RESULTS")
            print(f"{'='*60}")
            print(f"✓ Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"✓ CV Accuracy:    {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
            print(f"✓ CV Std Dev:     {np.std(cv_scores):.4f}")
            
            # SPEED BOOST 7: Skip detailed classification report
            # print(f"\n📋 Classification Report:")
            # print(classification_report(y_test_np, y_pred, zero_division=0))
            
            # SPEED BOOST 8: Generate only essential graphs
            print("\n📊 Generating visualizations...")
            start = time.time()
            graphs = self.generate_training_graphs_fast(
                y_test_np, y_pred, y_pred_proba, 
                y_train_np, accuracy, cv_scores, selector, X_train.columns
            )
            print(f"✓ Graphs generated: {time.time()-start:.2f}s")
            
            # Save model
            self.save_model()
            
            total_time = time.time() - total_start
            print(f"\n{'='*60}")
            print(f"✅ TOTAL TIME: {total_time:.2f}s")
            print(f"{'='*60}\n")
            
            techniques_used = [
                'Feature Selection (Top 30)',
                'Random Forest (100 trees)',
                'Class Weight Balancing',
                'Fast 3-Fold CV'
            ]
            
            result = {
                'accuracy': accuracy,
                'cv_accuracy': cv_accuracy,
                'cv_std': np.std(cv_scores),
                'features_used': len(available_features),
                'features_selected': n_features_to_select,
                'samples': len(df_clean),
                'samples_balanced': len(y_train_balanced),
                'feature_groups': self.feature_summary,
                'target_column': self.target_column,
                'classes': sorted(y.unique().tolist()),
                'graphs': graphs,
                'training_time': f"{total_time:.2f}s",
                'techniques_used': techniques_used
            }
            
            return result
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            traceback.print_exc()
            raise
    
    # Replace the train method with this simpler version for testing:
    def train_simple(self, file_path):
        """Simple training without advanced features (for debugging)"""
        try:
            print("\n=== SIMPLE TRAINING (DEBUG MODE) ===\n")
            
            # Load data
            print("Loading data...")
            df = pd.read_csv(file_path, low_memory=False)
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Find target
            possible_targets = ['NACCALZD', 'NACCUDSD', 'dx', 'CDRGLOB']
            target_col = None
            
            for t in possible_targets:
                if t in df.columns:
                    target_col = t
                    self.target_column = t
                    break
            
            if not target_col:
                raise ValueError(f"No target found. Available columns: {list(df.columns[:10])}")
            
            print(f"✓ Using target: {target_col}")
            
            # Get features
            features = self.select_features(df)
            print(f"✓ Found {len(features)} features")
            
            # Clean data
            df_clean = df.dropna(subset=[target_col])
            print(f"✓ Clean rows: {len(df_clean)}")
            
            # Prepare
            X = self.prepare_data(df_clean, features)
            y = df_clean[target_col].astype(int)
            
            self.encoded_feature_names = X.columns.tolist()
            print(f"✓ Prepared data: {X.shape}")
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train simple model
            print("\nTraining simple Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            print("✓ Training complete")
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            print(f"✓ Accuracy: {accuracy*100:.2f}%")
            
            # Save
            self.save_model()
            
            return {
                'accuracy': accuracy,
                'cv_accuracy': accuracy,
                'cv_std': 0.0,
                'features_used': len(features),
                'features_selected': len(features),
                'samples': len(df_clean),
                'samples_balanced': len(df_clean),
                'target_column': target_col,
                'classes': sorted(y.unique().tolist()),
                'feature_groups': self.feature_summary,
                'graphs': {},
                'training_time': '< 1 min',
                'techniques_used': ['Simple Random Forest']
            }
            
        except Exception as e:
            print(f"❌ Simple training failed: {str(e)}")
            traceback.print_exc()
            raise
    
    def generate_training_graphs_fast(self, y_test, y_pred, y_pred_proba, y_train, accuracy, cv_scores, selector, original_features):
        """Generate essential graphs quickly"""
        graphs = {}
        
        try:
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 70  # Even lower for speed
            
            # 1. Confusion Matrix only
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
            ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=70, bbox_inches='tight')
            buffer.seek(0)
            graphs['confusion_matrix'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # 2. Feature Importance - Top 10 only
            if hasattr(selector, 'scores_'):
                fig, ax = plt.subplots(figsize=(8, 5))
                feature_scores = pd.DataFrame({
                    'feature': original_features,
                    'score': selector.scores_
                }).sort_values('score', ascending=False).head(10)
                
                sns.barplot(data=feature_scores, x='score', y='feature', palette='viridis', ax=ax)
                ax.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
                ax.set_xlabel('Importance Score')
                ax.set_ylabel('Feature')
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=70, bbox_inches='tight')
                buffer.seek(0)
                graphs['feature_importance'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # 3. ROC Curve (binary only)
            if len(np.unique(y_test)) == 2:
                fig, ax = plt.subplots(figsize=(6, 5))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=70, bbox_inches='tight')
                buffer.seek(0)
                graphs['roc_curve'] = base64.b64encode(buffer.read()).decode()
                plt.close()
        
        except Exception as e:
            print(f"⚠️ Warning: Could not generate some graphs: {str(e)}")
    
        return graphs
    
    def generate_training_graphs_advanced(self, y_test, y_pred, y_pred_proba, y_train, accuracy, cv_scores, selector, original_features):
        """Generate advanced training visualization graphs"""
        graphs = {}
        
        try:
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 80
            
            # 1. Confusion Matrix with percentages
            fig, ax = plt.subplots(figsize=(7, 6))
            cm = confusion_matrix(y_test, y_pred)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            annot = np.array([[f'{val}\n({pct:.1f}%)' for val, pct in zip(row, pct_row)] 
                             for row, pct_row in zip(cm, cm_percent)])
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True, ax=ax)
            ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80)
            buffer.seek(0)
            graphs['confusion_matrix'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # 2. Selected Feature Importance
            if hasattr(selector, 'scores_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_scores = pd.DataFrame({
                    'feature': original_features,
                    'score': selector.scores_
                }).sort_values('score', ascending=False).head(15)
                
                sns.barplot(data=feature_scores, x='score', y='feature', palette='viridis', ax=ax)
                ax.set_title('Top 15 Selected Features (Mutual Information)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score')
                ax.set_ylabel('Feature')
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=80)
                buffer.seek(0)
                graphs['feature_importance'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # 3. ROC Curve
            if len(np.unique(y_test)) == 2:
                fig, ax = plt.subplots(figsize=(7, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                ax.fill_between(fpr, tpr, alpha=0.2, color='orange')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('ROC Curve - Ensemble Model', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=80)
                buffer.seek(0)
                graphs['roc_curve'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # 4. Cross-Validation Scores with statistics
            fig, ax = plt.subplots(figsize=(8, 5))
            folds = range(1, len(cv_scores) + 1)
            ax.plot(folds, cv_scores, marker='o', linestyle='-', linewidth=3, markersize=10, color='#667eea')
            ax.axhline(y=np.mean(cv_scores), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_scores):.3f}')
            ax.fill_between(folds, cv_scores, alpha=0.3, color='#667eea')
            ax.set_xlabel('Fold Number', fontsize=12)
            ax.set_ylabel('Accuracy Score', fontsize=12)
            ax.set_title(f'5-Fold Cross-Validation\nStd Dev: {np.std(cv_scores):.4f}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([max(0, min(cv_scores)-0.05), min(1, max(cv_scores)+0.05)])
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80)
            buffer.seek(0)
            graphs['cv_scores'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # 5. Class Distribution
            fig, ax = plt.subplots(figsize=(7, 5))
            class_counts = pd.Series(y_train).value_counts().sort_index()
            colors = ['#4caf50', '#f44336']
            bars = ax.bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(class_counts.index)
            
            for bar, (idx, val) in zip(bars, class_counts.items()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(class_counts.values)*0.02,
                       f'{val}\n({val/sum(class_counts.values)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80)
            buffer.seek(0)
            graphs['class_distribution'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Warning: Could not generate some graphs: {str(e)}")
            traceback.print_exc()
        
        return graphs
    
    def predict(self, input_data):
        """Make prediction with preprocessing"""
        try:
            if self.model is None:
                raise ValueError("Model not trained!")
            
            df = pd.DataFrame([input_data])
            
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            df = df[self.feature_names]
            X = self.prepare_data(df, self.feature_names)
            
            for col in self.encoded_feature_names:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.encoded_feature_names]
            X = X.fillna(0)
            
            # Apply scaling and feature selection
            if hasattr(self, 'scaler') and self.scaler is not None:
                X = self.scaler.transform(X)
            if hasattr(self, 'selector') and self.selector is not None:
                X = self.selector.transform(X)
            
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            return {
                'prediction': int(prediction),
                'confidence': float(max(probabilities)),
                'probabilities': {str(i): float(prob) for i, prob in enumerate(probabilities)}
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            traceback.print_exc()
            raise

    def predict_batch_optimized(self, df):
        """Batch prediction with detailed error handling"""
        try:
            if self.model is None:
                raise ValueError("Model not trained!")
            
            print(f"\n{'='*50}")
            print(f"PROCESSING {len(df)} ROWS")
            print(f"{'='*50}")
            
            # Add missing features
            print("Adding missing features...")
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            print(f"✓ Features prepared: {len(self.feature_names)}")
            
            # Select only needed features
            df_features = df[self.feature_names].copy()
            
            # Prepare data
            print("Preparing data...")
            try:
                X = self.prepare_data(df_features, self.feature_names)
                print(f"✓ Data prepared: {X.shape}")
            except Exception as e:
                print(f"❌ Error in prepare_data: {str(e)}")
                traceback.print_exc()
                raise
            
            # Add missing encoded features
            for col in self.encoded_feature_names:
                if col not in X.columns:
                    X[col] = 0
            
            # Reorder columns to match training
            X = X[self.encoded_feature_names]
            X = X.fillna(0)
            
            print(f"✓ Features aligned: {X.shape}")
            
            # Convert to numpy
            print("Converting to numpy...")
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            print(f"✓ Numpy array: {X_np.shape}")
            
            # Apply feature selection (if used during training)
            if hasattr(self, 'selector') and self.selector is not None:
                print("Applying feature selection...")
                try:
                    X_np = self.selector.transform(X_np)
                    print(f"✓ Features selected: {X_np.shape}")
                except Exception as e:
                    print(f"❌ Error in feature selection: {str(e)}")
                    traceback.print_exc()
                    raise
            
            # Make predictions
            print("Making predictions...")
            try:
                predictions = self.model.predict(X_np)
                print(f"✓ Predictions made: {len(predictions)}")
            except Exception as e:
                print(f"❌ Error in model.predict: {str(e)}")
                traceback.print_exc()
                raise
            
            # Get probabilities
            print("Getting probabilities...")
            try:
                probabilities = self.model.predict_proba(X_np)
                print(f"✓ Probabilities computed: {probabilities.shape}")
            except Exception as e:
                print(f"❌ Error in model.predict_proba: {str(e)}")
                traceback.print_exc()
                raise
            
            # Build results
            print("Building results...")
            results = []
            for i in range(len(predictions)):
                try:
                    results.append({
                        'prediction': int(predictions[i]),
                        'confidence': float(max(probabilities[i])),
                        'probabilities': {
                            str(j): float(probabilities[i][j]) 
                            for j in range(len(probabilities[i]))
                        }
                    })
                except Exception as e:
                    print(f"⚠️ Error processing row {i}: {str(e)}")
                    results.append({
                        'prediction': -1,
                        'confidence': 0.0,
                        'probabilities': {},
                        'error': str(e)
                    })
            
            print(f"✓ Results built: {len(results)}")
            print(f"{'='*50}\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ BATCH PREDICTION FAILED: {str(e)}")
            traceback.print_exc()
            raise
    
    def save_model(self):
        """Save the trained model with preprocessing"""
        if self.model is not None:
            os.makedirs('models', exist_ok=True)
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'encoded_feature_names': self.encoded_feature_names,
                'target_column': self.target_column,
                'scaler': getattr(self, 'scaler', None),
                'selector': getattr(self, 'selector', None)
            }
            joblib.dump(model_data, 'models/nacc_model.pkl')
            print("✓ Model saved with preprocessing")

    def load_model(self):
        """Load a previously trained model"""
        try:
            model_data = joblib.load('models/nacc_model.pkl')
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.encoded_feature_names = model_data['encoded_feature_names']
            self.target_column = model_data['target_column']
            self.scaler = model_data.get('scaler', None)
            self.selector = model_data.get('selector', None)
            print("✓ Model loaded with preprocessing")
            return True
        except FileNotFoundError:
            print("ℹ No saved model found")
            return False

# Create predictor instance
predictor = NACCPredictor()
predictor.load_model()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'OK',
        'message': 'Server is running',
        'model_trained': predictor.model is not None
    })

@app.route('/train', methods=['POST'])
def train():
    try:
        print("\n=== Train Endpoint Called ===")
        print("Request files:", request.files)
        
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        
        # Save with original filename (overwrite if exists)
        filepath = os.path.join('uploads', 'training_data.csv')
        
        print(f"Saving to: {filepath}")
        
        try:
            # Remove old file if exists
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print("✓ Removed old training file")
                except Exception as e:
                    print(f"⚠️ Could not remove old file: {str(e)}")
            
            # Save new file
            file.save(filepath)
            print(f"✓ File saved successfully")
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            return jsonify({'success': False, 'error': f'Cannot save file: {str(e)}'}), 500
        
        # Verify file exists and is readable
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File was not saved properly'}), 500
        
        file_size = os.path.getsize(filepath)
        print(f"✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        if file_size == 0:
            return jsonify({'success': False, 'error': 'Uploaded file is empty'}), 400
        
        # Train the model
        print("Starting training...")
        try:
            result = predictor.train(filepath)
            print("✓ Training completed successfully")
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            traceback.print_exc()
            
            return jsonify({
                'success': False,
                'error': f'Training failed: {str(e)}'
            }), 500
        
        # DON'T delete the file - keep it for future use
        print(f"✓ Training data saved at: {filepath}")
        
        # Build response
        response_data = {
            'success': True,
            'accuracy': float(result.get('accuracy', 0)),
            'cv_accuracy': float(result.get('cv_accuracy', 0)),
            'cv_std': float(result.get('cv_std', 0)),
            'features_used': int(result.get('features_used', 0)),
            'features_selected': int(result.get('features_selected', 0)),
            'samples': int(result.get('samples', 0)),
            'samples_balanced': int(result.get('samples_balanced', 0)),
            'target_column': str(result.get('target_column', 'Unknown')),
            'classes': [int(x) for x in result.get('classes', [])],
            'feature_groups': result.get('feature_groups', {}),
            'graphs': result.get('graphs', {}),
            'training_time': result.get('training_time', 'N/A'),
            'techniques_used': result.get('techniques_used', []),
            'data_file': filepath  # Include file path in response
        }
        
        print("✓ Sending response")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in train endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint with detailed debugging"""
    try:
        print("\n" + "="*60)
        print("BATCH PREDICTION STARTED")
        print("="*60)
        
        # Check if model is trained
        if predictor.model is None:
            print("❌ Model not trained!")
            return jsonify({'success': False, 'error': 'Model not trained! Please train the model first.'}), 400
        
        print("✓ Model is loaded")
        
        # Check file upload
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"✓ File received: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        
        # Save file
        filepath = os.path.join('uploads', 'prediction_data.csv')
        
        # Remove old file if exists
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print("✓ Removed old prediction file")
            except Exception as e:
                print(f"⚠️ Could not remove old file: {str(e)}")
                # Try with a different name
                import time as time_module
                filepath = os.path.join('uploads', f'prediction_data_{int(time_module.time())}.csv')
        
        # Save the uploaded file
        try:
            file.save(filepath)
            print(f"✓ File saved: {filepath}")
            
            # Verify file
            file_size = os.path.getsize(filepath)
            print(f"✓ File size: {file_size:,} bytes")
            
            if file_size == 0:
                return jsonify({'success': False, 'error': 'Uploaded file is empty'}), 400
                
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Cannot save file: {str(e)}'}), 500
        
        # Read CSV
        print("\n📂 Reading CSV file...")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"✓ Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        except Exception as e:
            print(f"❌ Error reading CSV: {str(e)}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Cannot read CSV: {str(e)}'}), 500
        
        if len(df) == 0:
            print("❌ CSV is empty")
            return jsonify({'success': False, 'error': 'CSV file contains no data rows'}), 400
        
        # Check if required features exist
        print("\n🔍 Checking features...")
        print(f"Model expects {len(predictor.feature_names)} features")
        print(f"CSV has {len(df.columns)} columns")
        
        missing_features = [f for f in predictor.feature_names if f not in df.columns]
        if missing_features:
            print(f"⚠️ Missing {len(missing_features)} features (will be set to 0):")
            print(f"  {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        
        # Make predictions
        print("\n🎯 Making predictions...")
        prediction_start = time.time()
        
        try:
            predictions = predictor.predict_batch_optimized(df)
            prediction_time = time.time() - prediction_start
            print(f"✓ Predictions completed in {prediction_time:.2f}s")
            print(f"✓ Processed {len(predictions)} records")
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'Prediction failed: {str(e)}',
                'details': traceback.format_exc()
            }), 500
        
        # Calculate statistics
        print("\n📊 Calculating statistics...")
        successful_predictions = [p for p in predictions if p.get('prediction', -1) != -1]
        
        if len(successful_predictions) == 0:
            print("❌ No successful predictions!")
            return jsonify({
                'success': False,
                'error': 'All predictions failed. Check if CSV columns match training data.'
            }), 500
        
        at_risk_count = sum(1 for p in successful_predictions if p['prediction'] == 1)
        not_at_risk_count = len(successful_predictions) - at_risk_count
        
        print(f"✓ At risk: {at_risk_count}")
        print(f"✓ Not at risk: {not_at_risk_count}")
        print(f"✓ Failed: {len(predictions) - len(successful_predictions)}")
        
        total_time = time.time() - prediction_start
        
        response_data = {
            'success': True,
            'total_predictions': len(predictions),
            'successful': len(successful_predictions),
            'failed': len(predictions) - len(successful_predictions),
            'at_risk': at_risk_count,
            'not_at_risk': not_at_risk_count,
            'predictions': predictions[:1000],  # Limit to 1000 for response size
            'processing_time': f"{total_time:.2f}s",
            'rows_per_second': int(len(df) / prediction_time) if prediction_time > 0 else 0,
            'data_file': filepath
        }
        
        print(f"\n{'='*60}")
        print("✅ BATCH PREDICTION COMPLETED")
        print(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ CRITICAL ERROR in predict-batch endpoint")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/files', methods=['GET'])
def list_files():
    """List all saved files"""
    try:
        files = []
        
        # Check uploads directory
        if os.path.exists('uploads'):
            for filename in os.listdir('uploads'):
                filepath = os.path.join('uploads', filename)
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    file_time = os.path.getmtime(filepath)
                    
                    files.append({
                        'name': filename,
                        'path': filepath,
                        'size': file_size,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
                    })
        
        # Check models directory
        if os.path.exists('models'):
            for filename in os.listdir('models'):
                filepath = os.path.join('models', filename)
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    file_time = os.path.getmtime(filepath)
                    
                    files.append({
                        'name': filename,
                        'path': filepath,
                        'size': file_size,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
                    })
        
        return jsonify({
            'success': True,
            'files': files,
            'total_files': len(files)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("NACC Disease Predictor Server")
    print("="*50)
    print("Server URL: http://127.0.0.1:5000")
    print("Model Status: " + ("Loaded" if predictor.model is not None else "Not trained"))
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)