import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import shap

class EmployeeAttritionModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        def ensure_string_columns(self, df):
            """Ensure all DataFrame column names are strings to avoid Streamlit warnings"""
            if not isinstance(df, pd.DataFrame):
                return df
            df = df.copy()
            df.columns = [str(col) for col in df.columns]
            return df
        
        self.df.columns = self.df.columns.astype(str)
        self.label_encoders = {}
        self.model_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
            "Support Vector Machine": SVC(C=200.0, kernel='rbf', probability=True),
            "Gaussian Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, bootstrap=True, oob_score=True ),
            "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.01, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=350, learning_rate=0.005, max_depth=3, random_state=42),
            "XGBoost": XGBClassifier(objective='binary:logistic', n_estimators=350, learning_rate=0.01, max_depth=3, random_state=42, eval_metric='logloss')
        }
        self.param_grids = {
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2', None]  # Fixed: changed 'auto' to valid options
            },
            "XGBoost": {
                'n_estimators': [100, 200, 350],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            },
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [5, 10, 15],
                'weights': ['uniform', 'distance']
            },
            "Support Vector Machine": {
                'C': [100, 200, 300],
                'kernel': ['rbf', 'linear']
            },
            "Decision Tree": {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }

        }
        self.fitted_models = {}
        self.scaler = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.right_skewed = None

    def get_random_forest_explanation(self):
        """
        Get a custom explanation for Random Forest as a fallback when SHAP fails
        
        Returns:
            Dictionary with feature importance data
        """
        if "Random Forest" not in self.fitted_models:
            return None
            
        model = self.fitted_models["Random Forest"]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return {
            'importances': importances,
            'indices': indices,
            'feature_names': [str(col) for col in self.X.columns]
        }
    
    def preprocess_data(self):
        self.df['Attrition'] = self.df['Attrition'].map({'Yes': 1, 'No': 0})
    
        self.df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True, errors='ignore')
    
        self.df['WorkExperience'] = self.df[['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                                             'YearsSinceLastPromotion', 'YearsWithCurrManager']].mean(axis=1)
    
        self.df['OverallSatisfaction'] = self.df[['JobSatisfaction', 'EnvironmentSatisfaction',
                                                  'RelationshipSatisfaction', 'WorkLifeBalance']].mean(axis=1)
    
        self.df.drop(['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                      'YearsSinceLastPromotion', 'JobSatisfaction', 'EnvironmentSatisfaction',
                      'RelationshipSatisfaction', 'WorkLifeBalance'], axis=1, inplace=True, errors='ignore')
    
        for col in self.df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
    
        numeric_df = self.df.drop('Attrition', axis=1)
        skewed_features = numeric_df.apply(lambda x: x.skew()).sort_values(ascending=False)
        self.right_skewed = skewed_features[skewed_features > 0.5].index
    
        for col in self.right_skewed:
            self.df[col] = np.log1p(self.df[col])
    
        self.X = self.df.drop('Attrition', axis=1)
        # Ensure column names are strings
        self.X.columns = [str(col) for col in self.X.columns]
        self.y = self.df['Attrition']

    def split_and_scale(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        sm = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = sm.fit_resample(self.X_train_scaled, self.y_train)

    def train_model(self, model_name, run_tuning=False):
        model = self.model_dict[model_name]

        if run_tuning and model_name in self.param_grids:
            grid = GridSearchCV(model, self.param_grids[model_name], cv=3, scoring='f1', n_jobs=-1)
            grid.fit(self.X_train_smote, self.y_train_smote)
            model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model.fit(self.X_train_smote, self.y_train_smote)
            best_params = None

        self.fitted_models[model_name] = model

        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "conf_matrix": confusion_matrix(self.y_test, y_pred),
            "classification_report": classification_report(self.y_test, y_pred),
            "probabilities": y_proba,
            "best_params": best_params
        }

    def get_feature_importance(self, model_name):
        """Get feature importance for a model with robust error handling"""
        model = self.fitted_models.get(model_name)
        if model is None:
            return None
    
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                return {
                    'importances': importances,
                    'indices': indices,
                    'feature_names': [str(col) for col in self.X.columns]
                }
    
            elif model_name == "Logistic Regression":
                coef = np.abs(model.coef_[0])
                indices = np.argsort(coef)[::-1]
                return {
                    'importances': coef,
                    'indices': indices,
                    'feature_names': [str(col) for col in self.X.columns]
                }
            return None
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None

    def get_shap_values(self, model_name):
        """
        Get SHAP values for a model with robust error handling
        
        Args:
            model_name: Name of the model to explain
            
        Returns:
            Dictionary with SHAP values and related data
        """
        if model_name not in self.fitted_models:
            return {
                'shap_values': None,
                'X_sample': None,
                'feature_names': [str(col) for col in self.X.columns],
                'use_feature_importance': True,
                'error': 'Model not fitted'
            }
        
        model = self.fitted_models[model_name]
        
        # Sample from test data to speed up SHAP calculation
        sample_size = min(100, len(self.X_test_scaled))
        X_sample = self.X_test.iloc[:sample_size].copy()
        X_sample_scaled = self.X_test_scaled[:sample_size].copy()
        
        # Convert feature names to list of strings for consistent handling
        feature_names = [str(col) for col in self.X.columns]
    
        # Use feature importance for Random Forest and Decision Tree to avoid SHAP issues
        if model_name in ["Random Forest", "Decision Tree"]:
            return {
                'shap_values': None,
                'X_sample': X_sample,
                'feature_names': feature_names,
                'use_feature_importance': True
            }
    
        try:
            # For XGBoost and other models
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model)
                # Get SHAP values
                shap_values = explainer(X_sample_scaled)
                    
            else:  # Other models like Gradient Boosting
                explainer = shap.TreeExplainer(model)
                try:
                    # Try newer API first
                    shap_values = explainer(X_sample_scaled)
                except:
                    # Fall back to older API if needed
                    shap_values = explainer.shap_values(X_sample_scaled)
                    # For classification models with two classes
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values = shap_values[1]  # Use positive class
                        
            return {
                'shap_values': shap_values,
                'X_sample': X_sample,
                'feature_names': feature_names
            }
            
        except Exception as e:
            print(f"Error calculating SHAP values for {model_name}: {str(e)}")
            return {
                'shap_values': None,
                'X_sample': X_sample,
                'feature_names': feature_names,
                'use_feature_importance': True,
                'error': str(e)
            }
        
    def get_feature_explanation(self, model_name):
        """
        Get comprehensive feature importance and examples for any model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with feature importance data
        """
        if model_name not in self.fitted_models:
            return None
            
        model = self.fitted_models[model_name]
        
        # Get feature importance
        importance_data = self.get_feature_importance(model_name)
        
        if importance_data is None:
            return None
            
        importances = importance_data['importances']
        indices = importance_data['indices']
        feature_names = [str(col) for col in self.X.columns]
        
        # Create top features table
        top_indices = indices[:10]  # Top 10 features
        top_features_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Importance': importances[top_indices]
        })
        
        # Sample a few examples
        sample_size = min(5, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        X_samples = self.X_test.iloc[sample_indices]
        
        # Make predictions
        X_scaled = self.scaler.transform(X_samples)
        predictions = model.predict(X_scaled)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = None
        
        # Create examples dataframe
        examples = []
        for i, idx in enumerate(sample_indices):
            example = {'Example': i+1}
            
            # Add prediction
            example['Predicted'] = 'Yes' if predictions[i] == 1 else 'No'
            if probabilities is not None:
                example['Probability'] = f"{probabilities[i]:.3f}"
                
            # Add top feature values
            for feat_idx in top_indices[:5]:
                feat_name = feature_names[feat_idx]
                example[feat_name] = X_samples.iloc[i][feat_name]
                
            examples.append(example)
            
        examples_df = pd.DataFrame(examples)
        
        # Return all data
        return {
            'top_features': top_features_df,
            'examples': examples_df,
            'importances': importances,
            'indices': indices,
            'feature_names': feature_names
        }


    def process_new_data(self, new_data):
        new_data = new_data.copy()
        new_data.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], inplace=True, errors='ignore')

        new_data['WorkExperience'] = new_data[['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                                               'YearsSinceLastPromotion', 'YearsWithCurrManager']].mean(axis=1)

        new_data['OverallSatisfaction'] = new_data[['JobSatisfaction', 'EnvironmentSatisfaction',
                                                    'RelationshipSatisfaction', 'WorkLifeBalance']].mean(axis=1)

        new_data.drop(['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                       'YearsSinceLastPromotion', 'JobSatisfaction', 'EnvironmentSatisfaction',
                       'RelationshipSatisfaction', 'WorkLifeBalance'], axis=1, inplace=True, errors='ignore')

        for col in new_data.select_dtypes(include='object').columns:
            if col in self.label_encoders:
                new_data[col] = self.label_encoders[col].transform(new_data[col])
            else:
                le = LabelEncoder()
                new_data[col] = le.fit_transform(new_data[col])

        for col in self.right_skewed:
            if col in new_data.columns:
                new_data[col] = np.log1p(new_data[col])

        for col in self.X.columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[self.X.columns]
        return new_data

    def predict(self, new_data, model_name):
        if model_name not in self.fitted_models:
            raise ValueError(f"{model_name} is not trained.")
        processed_data = self.process_new_data(new_data)
        processed_data.columns = processed_data.columns.astype(str)
        scaled_data = self.scaler.transform(processed_data)
        model = self.fitted_models[model_name]
        preds = model.predict(scaled_data)
        probs = model.predict_proba(scaled_data)[:, 1] if hasattr(model, "predict_proba") else None
        return {
            'predictions': preds,
            'probabilities': probs
        }
