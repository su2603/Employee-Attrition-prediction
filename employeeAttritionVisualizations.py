import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
import shap

class AttritionVisualizer:
    def __init__(self, model_instance):
        """
        Initialize the visualizer with an instance of EmployeeAttritionModel
        
        Args:
            model_instance: An instance of the EmployeeAttritionModel class
        """
        self.model = model_instance
        plt.style.use('ggplot')
        
    def set_plot_style(self):
        """Set consistent styling for all plots"""
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        sns.set_palette("Set2")
        
    # EDA Visualizations
    
    def plot_attrition_distribution(self):
        """Visualize the distribution of attrition in the dataset"""
        self.set_plot_style()
        attrition_counts = self.model.df['Attrition'].value_counts()
        ax = sns.countplot(x=self.model.df['Attrition'])
        
        # Add percentage labels
        total = len(self.model.df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                        ha='center', fontsize=12)
        
        plt.title('Employee Attrition Distribution', fontsize=16)
        plt.xlabel('Attrition (1=Yes, 0=No)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, numerical_features=None):
        """
        Plot distribution of numerical features
        
        Args:
            numerical_features: List of numerical feature names to plot (if None, selects top 6)
        """
        if numerical_features is None:
            numerical_features = self.model.X.select_dtypes(include=['int64', 'float64']).columns[:6]
        
        self.set_plot_style()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                sns.histplot(data=self.model.df, x=feature, hue='Attrition', 
                             multiple="stack", kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}', fontsize=14)
                axes[i].set_xlabel(feature, fontsize=12)
                axes[i].set_ylabel('Count', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Plot correlation matrix heatmap of features"""
        self.set_plot_style()
        plt.figure(figsize=(16, 14))
        
        # Calculate correlation matrix
        corr = self.model.X.corr()
        
        # Generate mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot the heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_by_attrition(self, categorical_features=None):
        """
        Plot categorical features grouped by attrition
        
        Args:
            categorical_features: List of categorical feature names to plot (if None, auto-selects)
        """
        if categorical_features is None:
            # Select original categorical features before encoding
            categorical_features = list(self.model.label_encoders.keys())[:4]
        
        self.set_plot_style()
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        axes = axes.flatten()
        
        # Create a temporary dataframe with decoded categorical values
        temp_df = self.model.df.copy()
        temp_df['Attrition'] = temp_df['Attrition'].map({1: 'Yes', 0: 'No'})
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                if feature in self.model.label_encoders:
                    # Reverse the label encoding for better visualization
                    encoder = self.model.label_encoders[feature]
                    temp_df[feature] = encoder.inverse_transform(temp_df[feature])
                
                # Calculate proportions
                props = temp_df.groupby([feature, 'Attrition']).size().unstack().fillna(0)
                props = props.div(props.sum(axis=1), axis=0)
                
                # Plot
                props.plot(kind='bar', stacked=True, ax=axes[i])
                axes[i].set_title(f'{feature} by Attrition', fontsize=14)
                axes[i].set_xlabel(feature, fontsize=12)
                axes[i].set_ylabel('Proportion', fontsize=12)
                axes[i].legend(title='Attrition')
        
        plt.tight_layout()
        plt.show()

    # Model Performance Visualizations
    
    def plot_roc_curves(self, model_results):
        """
        Plot ROC curves for multiple models
        
        Args:
            model_results: Dictionary with model names as keys and results (containing 'probabilities') as values
        """
        self.set_plot_style()
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            if 'probabilities' in results and results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.model.y_test, results['probabilities'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self, model_results):
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            model_results: Dictionary with model names as keys and results (containing 'probabilities') as values
        """
        self.set_plot_style()
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            if 'probabilities' in results and results['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(self.model.y_test, results['probabilities'])
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name, confusion_mat):
        """
        Plot confusion matrix for a model
        
        Args:
            model_name: Name of the model
            confusion_mat: Confusion matrix from model results
        """
        self.set_plot_style()
        plt.figure(figsize=(8, 6))
        
        # Convert to dataframe for better visualization
        cm_df = pd.DataFrame(confusion_mat, 
                             index=['Actual Negative', 'Actual Positive'], 
                             columns=['Predicted Negative', 'Predicted Positive'])
        
        # Calculate percentages
        total = np.sum(confusion_mat)
        cm_norm = confusion_mat / total
        
        # Plot
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                plt.text(j+0.5, i+0.7, f'({cm_norm[i, j]:.1%})', 
                        ha='center', va='center', fontsize=12)
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, model_results):
        """
        Compare multiple models based on performance metrics
        
        Args:
            model_results: Dictionary with model names as keys and results as values
        """
        self.set_plot_style()
        
        # Extract metrics for all models
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(model_results.keys())
        
        # Create dataframe for easier plotting
        comparison_data = []
        for model_name, results in model_results.items():
            model_metrics = {metric: results[metric] for metric in metrics}
            model_metrics['Model'] = model_name
            comparison_data.append(model_metrics)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reshape for plotting
        comparison_plot_df = pd.melt(comparison_df, id_vars=['Model'], 
                                     value_vars=metrics, var_name='Metric', value_name='Score')
        
        # Plot
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_plot_df)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric', fontsize=12)
        plt.tight_layout()
        plt.show()

    # Feature Importance Visualizations
    
    def plot_feature_importance(self, model_name):
        """
        Plot feature importances for a model
        
        Args:
            model_name: Name of the model
        """
        importance_data = self.model.get_feature_importance(model_name)
        
        if importance_data is None:
            print(f"No feature importance available for {model_name}")
            return
            
        importances = importance_data['importances']
        indices = importance_data['indices']
        feature_names = importance_data['feature_names']
            
        self.set_plot_style()
        plt.figure(figsize=(12, 10))
            
        # Sort by importance
        sorted_indices = indices[:15]  # Top 15 features
        sorted_importances = importances[sorted_indices]
        sorted_features = feature_names[sorted_indices]
            
        # Plot horizontal bar chart
        plt.barh(range(len(sorted_indices)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_indices)), sorted_features)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.title(f'Top 15 Features - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
            
    def plot_permutation_importance(self, model_name, n_repeats=10):
        """
        Plot permutation importance for a model
        
        Args:
            model_name: Name of the model
            n_repeats: Number of times to permute each feature
        """
        if model_name not in self.model.fitted_models:
            print(f"Model {model_name} not found")
            return
            
        result = permutation_importance(
            self.model.fitted_models[model_name], self.model.X_test_scaled, 
            self.model.y_test, n_repeats=n_repeats, random_state=42
        )
            
        sorted_idx = result.importances_mean.argsort()[-15:]  # Top 15 features
            
        self.set_plot_style()
        plt.figure(figsize=(12, 10))
        plt.boxplot(result.importances[sorted_idx].T, 
                   vert=False, labels=self.model.X.columns[sorted_idx])
        plt.title(f"Permutation Importance - {model_name}", fontsize=16)
        plt.xlabel("Decrease in Accuracy", fontsize=14)
        plt.tight_layout()
        plt.show()
            
    def plot_shap_summary(self, model_name):
        """
        Plot SHAP summary for a model
        
        Args:
            model_name: Name of the model
        """
        shap_data = self.model.get_shap_values(model_name)
            
        if shap_data is None:
            print(f"No SHAP values available for {model_name}")
            return
            
        shap_values = shap_data["shap_values"]
        X_sample = shap_data["X_sample"]
            
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
            
    def plot_shap_dependence(self, model_name, feature_name):
        """
        Plot SHAP dependence for a specific feature
        
        Args:
            model_name: Name of the model
            feature_name: Name of the feature to analyze
        """
        shap_data = self.model.get_shap_values(model_name)
            
        if shap_data is None:
            print(f"No SHAP values available for {model_name}")
            return
            
        if feature_name not in shap_data["feature_names"]:
            print(f"Feature {feature_name} not found")
            return
            
        shap_values = shap_data["shap_values"]
        X_sample = shap_data["X_sample"]
            
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature_name, shap_values.values, X_sample, 
                           feature_names=shap_data["feature_names"], show=False)
        plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=16)
        plt.tight_layout()
        plt.show()

    # Class Imbalance Visualizations
    
    def plot_smote_comparison(self):
        """Plot the effect of SMOTE on class distribution"""
        self.set_plot_style()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        # Before SMOTE
        sns.countplot(x=self.model.y_train, ax=axes[0])
        axes[0].set_title('Class Distribution Before SMOTE', fontsize=14)
        axes[0].set_xlabel('Attrition (1=Yes, 0=No)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total_before = len(self.model.y_train)
        for p in axes[0].patches:
            percentage = f'{100 * p.get_height() / total_before:.1f}%'
            axes[0].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                          ha='center', fontsize=12)
        
        # After SMOTE
        sns.countplot(x=self.model.y_train_smote, ax=axes[1])
        axes[1].set_title('Class Distribution After SMOTE', fontsize=14)
        axes[1].set_xlabel('Attrition (1=Yes, 0=No)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total_after = len(self.model.y_train_smote)
        for p in axes[1].patches:
            percentage = f'{100 * p.get_height() / total_after:.1f}%'
            axes[1].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                          ha='center', fontsize=12)
            
        plt.tight_layout()
        plt.show()
        
    # Employee Risk Profiling Visualizations
    
    def plot_risk_by_department(self, model_name, original_data):
        """
        Plot attrition risk by department
        
        Args:
            model_name: Name of the model to use for predictions
            original_data: Original dataframe with 'Department' column
        """
        if model_name not in self.model.fitted_models:
            print(f"Model {model_name} not found")
            return
            
        # Make predictions on the data
        processed_data = self.model.process_new_data(original_data)
        scaled_data = self.model.scaler.transform(processed_data)
        model = self.model.fitted_models[model_name]
        
        # Get probabilities of attrition
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_data)[:, 1]
        else:
            print(f"Model {model_name} doesn't support probability predictions")
            return
            
        # Create dataframe with departments and risk scores
        risk_df = pd.DataFrame({
            'Department': original_data['Department'],
            'AttritionRisk': probs
        })
            
        self.set_plot_style()
        plt.figure(figsize=(14, 8))
            
        # Calculate mean risk by department
        dept_risk = risk_df.groupby('Department')['AttritionRisk'].mean().sort_values(ascending=False)
            
        # Plot
        ax = sns.barplot(x=dept_risk.index, y=dept_risk.values)
        plt.title('Average Attrition Risk by Department', fontsize=16)
        plt.xlabel('Department', fontsize=14)
        plt.ylabel('Average Attrition Risk Probability', fontsize=14)
        plt.xticks(rotation=45, ha='right')
            
        # Add percentage labels
        for i, v in enumerate(dept_risk.values):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12)
            
        plt.tight_layout()
        plt.show()
            
    def plot_risk_heatmap(self, model_name, original_data, feature1='MonthlyIncome', feature2='Age'):
        """
        Create a heatmap of attrition risk by two features
        
        Args:
            model_name: Name of the model to use for predictions
            original_data: Original dataframe
            feature1: First feature for the heatmap (x-axis)
            feature2: Second feature for the heatmap (y-axis)
        """
        if model_name not in self.model.fitted_models:
            print(f"Model {model_name} not found")
            return
            
        if feature1 not in original_data.columns or feature2 not in original_data.columns:
            print(f"Features {feature1} or {feature2} not found in data")
            return
            
        # Make predictions on the data
        processed_data = self.model.process_new_data(original_data)
        scaled_data = self.model.scaler.transform(processed_data)
        model = self.model.fitted_models[model_name]
            
        # Get probabilities of attrition
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_data)[:, 1]
        else:
            print(f"Model {model_name} doesn't support probability predictions")
            return
            
        # Create dataframe with features and risk scores
        risk_df = pd.DataFrame({
            feature1: original_data[feature1],
            feature2: original_data[feature2],
            'AttritionRisk': probs
        })
            
        # Create pivot table for heatmap
        heatmap_data = risk_df.pivot_table(
            values='AttritionRisk', 
            index=pd.qcut(risk_df[feature2], 10), 
            columns=pd.qcut(risk_df[feature1], 10), 
            aggfunc='mean'
        )
            
        self.set_plot_style()
        plt.figure(figsize=(14, 10))
            
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f'Attrition Risk Heatmap by {feature1} and {feature2}', fontsize=16)
        plt.tight_layout()
        plt.show()
            
    def plot_learning_curve(self, model_name, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Plot learning curve for a model
        
        Args:
            model_name: Name of the model
            cv: Number of cross-validation folds
            train_sizes: Array of training set sizes to evaluate
        """
        from sklearn.model_selection import learning_curve
            
        if model_name not in self.model.fitted_models:
            print(f"Model {model_name} not found")
            return
            
        model = self.model.fitted_models[model_name]
            
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.model.X_train_smote, self.model.y_train_smote,
            train_sizes=train_sizes, cv=cv, scoring='f1', n_jobs=-1
        )
            
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
            
        self.set_plot_style()
        plt.figure(figsize=(10, 8))
            
        # Plot means
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
            
        # Plot standard deviations
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='g')
            
        plt.title(f'Learning Curve - {model_name}', fontsize=16)
        plt.xlabel('Training Set Size', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage in the main file:
"""
def main():
    # Create and train the model
    model = EmployeeAttritionModel("HR_data.csv")
    model.preprocess_data()
    model.split_and_scale()
    
    # Train models
    model_results = {}
    for model_name in ["Random Forest", "XGBoost", "Logistic Regression"]:
        results = model.train_model(model_name)
        model_results[model_name] = results
    
    # Initialize visualizer
    viz = AttritionVisualizer(model)
    
    # Example visualizations
    viz.plot_attrition_distribution()
    viz.plot_feature_distributions()
    viz.plot_correlation_heatmap()
    viz.plot_categorical_by_attrition()
    
    viz.plot_roc_curves(model_results)
    viz.plot_precision_recall_curves(model_results)
    viz.plot_confusion_matrix("Random Forest", model_results["Random Forest"]["conf_matrix"])
    viz.plot_model_comparison(model_results)
    
    viz.plot_feature_importance("Random Forest")
    viz.plot_permutation_importance("Random Forest")
    viz.plot_shap_summary("XGBoost")
    viz.plot_shap_dependence("XGBoost", "MonthlyIncome")
    
    viz.plot_smote_comparison()
    
    # For risk profiling, we need the original data
    original_data = pd.read_csv("HR_data.csv")
    viz.plot_risk_by_department("XGBoost", original_data)
    viz.plot_risk_heatmap("XGBoost", original_data)
    
    viz.plot_learning_curve("Random Forest")

if __name__ == "__main__":
    main()
"""
