import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from employeeAttritionPred import EmployeeAttritionModel
from employeeAttritionVisualizations import AttritionVisualizer

def ensure_string_columns(df):
    """Ensure all DataFrame column names are strings to avoid Streamlit warnings"""
    if not isinstance(df, pd.DataFrame):
        return df
    df = df.copy()
    df.columns = df.columns.astype(str)
    return df

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

@st.cache_resource
def load_model():
    """Load and initialize the model (cached)"""
    try:
        model = EmployeeAttritionModel("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        model.preprocess_data()
        model.split_and_scale()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Employee Attrition Prediction with SHAP, Hyperparameter Tuning, and Batch Prediction")
    
    # Initialize model
    model = load_model()
    if model is None:
        st.stop()
    
    # Initialize visualizer
    visualizer = AttritionVisualizer(model)
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Model Training & Evaluation", "Data Visualizations", "Batch Prediction"]
    )
    
    if page == "Model Training & Evaluation":
        model_training_page(model, visualizer)
    elif page == "Data Visualizations":
        data_visualization_page(model, visualizer)
    elif page == "Batch Prediction":
        batch_prediction_page(model)

def model_training_page(model, visualizer):
    st.header("Model Training and Evaluation")
    
    model_options = list(model.model_dict.keys())
    
    selected_models = st.multiselect(
        "Select Models to Evaluate", 
        model_options, 
        default=["Random Forest", "XGBoost"]
    )
    
    run_tuning = st.checkbox("Run hyperparameter tuning (GridSearchCV) (select exactly one model)")
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Check if tuning is selected but multiple models are chosen
    if run_tuning and len(selected_models) > 1:
        st.warning("Please select only one model when running hyperparameter tuning.")
        return
    
    # Train button
    if st.button("Train Selected Models"):
        # Progress bar for model training
        progress_bar = st.progress(0)
        model_scores = {}
        model_results = {}
        
        # Train and evaluate models
        for i, model_name in enumerate(selected_models):
            with st.spinner(f"Training {model_name}..."):
                metrics = model.train_model(model_name, run_tuning=run_tuning)
                model_results[model_name] = metrics
                
                # Store scores for comparison
                model_scores[model_name] = {
                    'Accuracy': metrics['accuracy'], 
                    'Precision': metrics['precision'], 
                    'Recall': metrics['recall'], 
                    'F1': metrics['f1']
                }
                
                # Display results
                st.subheader(f"{model_name} Results")
                
                if metrics['best_params']:
                    st.write(f"Best Parameters: {metrics['best_params']}")
                
                st.text("Classification Report:")
                st.text(metrics['classification_report'])
                
                # Plot confusion matrix
                fig, ax = plt.subplots()
                sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix: {model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close(fig)
                
                # Show SHAP values for tree-based models
                if model_name in ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree"]:
                    if model_name in ["Random Forest", "Decision Tree"]:
                        st.info(f"SHAP values not available for {model_name}. Showing feature importance instead.")
                        
                        # Display feature importance
                        importance_data = model.get_feature_importance(model_name)
                        if importance_data:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            plt.title(f'Feature Importances ({model_name})')
                            plt.barh(
                                range(len(importance_data['indices'][:15])), 
                                importance_data['importances'][importance_data['indices'][:15]], 
                                align='center'
                            )
                            plt.yticks(
                                range(len(importance_data['indices'][:15])), 
                                [importance_data['feature_names'][i] for i in importance_data['indices'][:15]]
                            )
                            plt.xlabel('Relative Importance')
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        shap_data = model.get_shap_values(model_name)
                        if shap_data is not None and shap_data.get('shap_values') is not None:
                            st.write(f"SHAP summary plot for {model_name} (sample of test data):")
                            
                            try:
                                # Create a new figure explicitly
                                fig = plt.figure(figsize=(10, 8))
                                plt.clf()  # Clear the figure
                                
                                # Get feature names as strings
                                feature_names = [str(name) for name in shap_data['feature_names']]
                                
                                # Plot SHAP summary
                                if hasattr(shap_data['shap_values'], 'values'):
                                    # Newer SHAP API
                                    shap.summary_plot(
                                        shap_data['shap_values'], 
                                        shap_data['X_sample'], 
                                        show=False,
                                        plot_size=(10, 8)
                                    )
                                else:
                                    # Older SHAP API
                                    shap.summary_plot(
                                        shap_data['shap_values'], 
                                        shap_data['X_sample'].values, 
                                        feature_names=feature_names,
                                        show=False,
                                        plot_size=(10, 8)
                                    )
                                
                                # Render the plot
                                st.pyplot(plt.gcf())
                                plt.close()
                            except Exception as e:
                                st.error(f"Error creating SHAP plot: {str(e)}")
                                # Show feature importance as fallback
                                importance_data = model.get_feature_importance(model_name)
                                if importance_data:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    plt.title(f'Feature Importances ({model_name})')
                                    plt.barh(
                                        range(len(importance_data['indices'][:15])), 
                                        importance_data['importances'][importance_data['indices'][:15]], 
                                        align='center'
                                    )
                                    plt.yticks(
                                        range(len(importance_data['indices'][:15])), 
                                        [importance_data['feature_names'][i] for i in importance_data['indices'][:15]]
                                    )
                                    plt.xlabel('Relative Importance')
                                    st.pyplot(fig)
                                    plt.close(fig)

            # Update progress bar
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Show model comparison
        if model_scores:
            st.subheader("Model Comparison")
            metrics_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={'index': 'Model'})
            metrics_df = ensure_string_columns(metrics_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.plot(kind='bar', x='Model', ax=ax)
            plt.title("Model Performance Metrics")
            plt.ylabel("Score")
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
            
            # ROC Curves comparison
            if len(selected_models) > 1:
                st.subheader("ROC Curves Comparison")
                fig = plt.figure(figsize=(10, 6))
                for model_name, results in model_results.items():
                    if 'probabilities' in results and results['probabilities'] is not None:
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(model.y_test, results['probabilities'])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)
        
        # Feature Importance
        if model.fitted_models:
            st.header("Feature Importance")
            
            model_for_importance = st.selectbox(
                "Select model for feature importance", 
                list(model.fitted_models.keys()),
                index=0 if "Random Forest" in model.fitted_models else 0
            )
            
            importance_data = model.get_feature_importance(model_for_importance)
            
            if importance_data:
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.title(f'Feature Importances ({model_for_importance})')
                plt.barh(
                    range(len(importance_data['indices'])), 
                    importance_data['importances'][importance_data['indices']], 
                    align='center'
                )
                plt.yticks(
                    range(len(importance_data['indices'])), 
                    [importance_data['feature_names'][i] for i in importance_data['indices']]
                )
                plt.xlabel('Relative Importance')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.write("Feature importance not available for this model type")

def data_visualization_page(model, visualizer):
    st.header("Data Visualizations and Insights")
    
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Exploratory Data Analysis", "Model Performance", "Feature Importance", "SHAP Visualizations"]
    )
    
    if viz_type == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        
        # Attrition Distribution
        st.write("### Attrition Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        attrition_counts = model.df['Attrition'].value_counts()
        sns.countplot(x=model.df['Attrition'], ax=ax)
        # Add percentage labels
        total = len(model.df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 5), ha='center')
        ax.set_title('Employee Attrition Distribution')
        ax.set_xlabel('Attrition (1=Yes, 0=No)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature Distributions
        st.write("### Numerical Feature Distributions")
        numerical_features = st.multiselect(
            "Select numerical features to visualize",
            model.X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            ["WorkExperience", "OverallSatisfaction", "MonthlyIncome", "Age"] if "MonthlyIncome" in model.X.columns else ["WorkExperience", "OverallSatisfaction"]
        )
        
        if numerical_features:
            fig, axes = plt.subplots(nrows=len(numerical_features), figsize=(12, 4*len(numerical_features)))
            if len(numerical_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(numerical_features):
                sns.histplot(data=model.df, x=feature, hue='Attrition', multiple="stack", kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Correlation Matrix
        st.write("### Feature Correlation Heatmap")
        corr = model.X.corr()
        corr = ensure_string_columns(pd.DataFrame(corr)) 

        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax)
        plt.title('Feature Correlation Heatmap')
        st.pyplot(fig)
        plt.close(fig)
        
        # SMOTE Comparison
        st.write("### Class Distribution Before and After SMOTE")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        
        # Before SMOTE
        sns.countplot(x=model.y_train, ax=axes[0])
        axes[0].set_title('Before SMOTE')
        axes[0].set_xlabel('Attrition')
        
        # After SMOTE
        sns.countplot(x=model.y_train_smote, ax=axes[1])
        axes[1].set_title('After SMOTE')
        axes[1].set_xlabel('Attrition')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    elif viz_type == "Model Performance":
        st.subheader("Model Performance Visualizations")
        
        if not model.fitted_models:
            st.warning("No trained models available. Please train models first in the 'Model Training & Evaluation' page.")
            return
        
        # Model Comparison
        st.write("### Model Performance Comparison")
        
        model_results = {}
        for model_name in model.fitted_models.keys():
            # We need to make predictions to get metrics for the plot
            y_pred = model.fitted_models[model_name].predict(model.X_test_scaled)
            y_proba = model.fitted_models[model_name].predict_proba(model.X_test_scaled)[:, 1] if hasattr(model.fitted_models[model_name], "predict_proba") else None
            
            model_results[model_name] = {
                "accuracy": accuracy_score(model.y_test, y_pred),
                "precision": precision_score(model.y_test, y_pred),
                "recall": recall_score(model.y_test, y_pred),
                "f1": f1_score(model.y_test, y_pred),
                "probabilities": y_proba
            }
        
        # Create dataframe for model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        comparison_data = []
        for model_name, results in model_results.items():
            model_metrics = {metric: results[metric] for metric in metrics}
            model_metrics['Model'] = model_name
            comparison_data.append(model_metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_plot_df = pd.melt(comparison_df, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_plot_df, ax=ax)
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # ROC Curves
        st.write("### ROC Curves Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        from sklearn.metrics import roc_curve, auc
        for model_name, results in model_results.items():
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(model.y_test, results['probabilities'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        st.pyplot(fig)
        plt.close(fig)
        
        # Precision-Recall Curves
        st.write("### Precision-Recall Curves Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        from sklearn.metrics import precision_recall_curve
        for model_name, results in model_results.items():
            if results['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(model.y_test, results['probabilities'])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        st.pyplot(fig)
        plt.close(fig)
        
    elif viz_type == "Feature Importance":
        st.subheader("Feature Importance Analysis")
        
        if not model.fitted_models:
            st.warning("No trained models available. Please train models first in the 'Model Training & Evaluation' page.")
            return
        
        # Select model for feature importance
        model_for_importance = st.selectbox(
            "Select model for feature importance analysis", 
            list(model.fitted_models.keys()),
            index=0 if "Random Forest" in model.fitted_models else 0
        )
        
        # Display feature importance
        importance_data = model.get_feature_importance(model_for_importance)
        
        if importance_data:
            st.write(f"### Feature Importance for {model_for_importance}")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Sort by importance and show top features
            sorted_indices = importance_data['indices'][:15]  # Top 15 features
            sorted_importances = importance_data['importances'][sorted_indices]
            sorted_features = [importance_data['feature_names'][i] for i in sorted_indices]
            
            # Plot horizontal bar chart
            plt.barh(range(len(sorted_indices)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_indices)), sorted_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Features - {model_for_importance}')
            st.pyplot(fig)
            plt.close(fig)
            
            # Show permutation importance if desired
            if st.checkbox("Show Permutation Importance (may take time to compute)"):
                with st.spinner("Computing permutation importance..."):
                    from sklearn.inspection import permutation_importance
                    
                    result = permutation_importance(
                        model.fitted_models[model_for_importance], 
                        model.X_test_scaled, 
                        model.y_test, 
                        n_repeats=5, 
                        random_state=42
                    )
                    
                    sorted_idx = result.importances_mean.argsort()[-15:]  # Top 15 features
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    plt.boxplot(result.importances[sorted_idx].T, 
                               vert=False, labels=[model.X.columns[i] for i in sorted_idx])
                    plt.title(f"Permutation Importance - {model_for_importance}")
                    plt.xlabel("Decrease in Accuracy")
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.write("Feature importance not available for this model type")

    elif viz_type == "SHAP Visualizations":
        st.subheader("SHAP Value Analysis")
        
        if not model.fitted_models:
            st.warning("No trained models available. Please train models first in the 'Model Training & Evaluation' page.")
            return
        
        # Get tree-based models
        tree_models = [m for m in model.fitted_models.keys() 
                       if m in ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree"]]
        
        if not tree_models:
            st.warning("No tree-based models available for SHAP analysis. Please train Random Forest, XGBoost, Gradient Boosting, or Decision Tree models.")
            return
            
        # Select model for SHAP analysis
        model_for_shap = st.selectbox(
            "Select model for SHAP analysis", 
            tree_models,
            index=0
        )
        
        # Generate SHAP values
        shap_data = model.get_shap_values(model_for_shap)
        
        if shap_data is not None:
            # Check if we should use feature importance instead of SHAP
            if 'use_feature_importance' in shap_data and shap_data['use_feature_importance']:
                st.info(f"SHAP visualization not available for {model_for_shap}. Showing feature importance instead.")
                
                # Get and display feature explanation
                explanation = model.get_feature_explanation(model_for_shap)
                if explanation:
                    # Show feature importance
                    st.write("### Top Important Features")
                    st.dataframe(explanation['top_features'])
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_n = 15
                    indices = explanation['indices'][:top_n]
                    
                    ax.barh(range(len(indices)), 
                            explanation['importances'][indices], 
                            align='center')
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([explanation['feature_names'][i] for i in indices])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'Top {top_n} Features - {model_for_shap}')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show example predictions
                    st.write("### Example Predictions")
                    st.write("The table below shows example predictions with values for top features:")
                    st.dataframe(explanation['examples'])
                    
            # If we have valid SHAP values, display them
            elif shap_data['shap_values'] is not None:
                # Display SHAP summary plot
                st.write(f"### SHAP Summary Plot for {model_for_shap}")
                
                try:
                    # Create figure for SHAP summary plot
                    fig = plt.figure(figsize=(12, 10))
                    plt.clf()
                    
                    # Convert feature names to list of strings
                    feature_names = [str(name) for name in shap_data['feature_names']]
                    
                    # Handle different SHAP value types
                    if hasattr(shap_data['shap_values'], 'values'):
                        # For newer SHAP API (Explanation objects)
                        shap.summary_plot(
                            shap_data['shap_values'], 
                            shap_data['X_sample'],
                            plot_size=(12, 10),
                            show=False
                        )
                    else:
                        # For older SHAP API (numpy arrays)
                        shap.summary_plot(
                            shap_data['shap_values'], 
                            shap_data['X_sample'].values,
                            feature_names=feature_names,
                            plot_size=(12, 10),
                            show=False
                        )
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # SHAP Dependence Plots for Top Features
                    st.write("### SHAP Dependence Plots")
                    
                    # Calculate mean absolute SHAP values to find top features
                    if hasattr(shap_data['shap_values'], 'values'):
                        # For newer SHAP API
                        shap_values_array = shap_data['shap_values'].values
                        # Calculate mean absolute values
                        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
                    else:
                        # For older SHAP API
                        mean_abs_shap = np.abs(shap_data['shap_values']).mean(axis=0)
                    
                    # Get top 3 feature indices
                    top_indices = np.argsort(mean_abs_shap)[-3:]
                    
                    # Create dependence plots for top features
                    for idx in top_indices:
                        if idx < len(feature_names):
                            feature = feature_names[idx]
                        else:
                            feature = f"Feature_{idx}"
                        
                        st.write(f"#### Dependence Plot for {feature}")
                        
                        # Create dependence plot
                        fig = plt.figure(figsize=(10, 6))
                        
                        # Get feature values and SHAP values
                        if idx < shap_data['X_sample'].shape[1]:
                            x = shap_data['X_sample'].iloc[:, idx].values
                            
                            if hasattr(shap_data['shap_values'], 'values'):
                                y = shap_data['shap_values'].values[:, idx]
                            else:
                                y = shap_data['shap_values'][:, idx]
                            
                            # Create scatter plot manually
                            plt.scatter(x, y, alpha=0.7)
                            plt.xlabel(feature)
                            plt.ylabel(f'SHAP value for {feature}')
                            plt.title(f'SHAP Dependence Plot for {feature}')
                            plt.grid(alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close(fig)
                
                except Exception as e:
                    st.error(f"Error generating SHAP plots: {str(e)}")
                    st.info("Showing feature importance instead...")
                    
                    # Show feature importance as fallback
                    explanation = model.get_feature_explanation(model_for_shap)
                    if explanation:
                        # Show feature importance
                        st.write("### Top Important Features")
                        st.dataframe(explanation['top_features'])
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(12, 8))
                        top_n = 15
                        indices = explanation['indices'][:top_n]
                        
                        ax.barh(range(len(indices)), 
                                explanation['importances'][indices], 
                                align='center')
                        ax.set_yticks(range(len(indices)))
                        ax.set_yticklabels([explanation['feature_names'][i] for i in indices])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title(f'Top {top_n} Features - {model_for_shap}')
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Show example predictions
                        st.write("### Example Predictions")
                        st.write("The table below shows example predictions with values for top features:")
                        st.dataframe(explanation['examples'])
            else:
                st.warning("SHAP values could not be calculated for this model.")
        else:
            st.error("Failed to generate model explanation.")

def batch_prediction_page(model):
    st.header("Batch Prediction: Upload CSV")
    
    uploaded_file = st.file_uploader("Upload CSV with employee data for batch prediction", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(ensure_string_columns(new_data.head()))
            
            # Check for required columns
            required_columns = [
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'JobSatisfaction', 'EnvironmentSatisfaction',
                'RelationshipSatisfaction', 'WorkLifeBalance'
            ]
            
            missing_req_columns = [col for col in required_columns if col not in new_data.columns]
            if missing_req_columns:
                st.error(f"Missing required columns: {', '.join(missing_req_columns)}")
                return
            
            # Select model for prediction
            pred_model_options = list(model.fitted_models.keys())
            if not pred_model_options:
                st.warning("No trained models available. Please train a model first.")
                return
                
            pred_model_name = st.selectbox(
                "Select model for prediction", 
                pred_model_options,
                index=0 if "Random Forest" in pred_model_options else 0
            )
            
            # Make predictions
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    try:
                        prediction_results = model.predict(new_data, pred_model_name)
                        
                        # Get predictions and probabilities
                        preds = prediction_results['predictions']
                        pred_probs = prediction_results['probabilities']
                        
                        # Display results
                        results_df = pd.DataFrame(new_data)
                        results_df.columns = [str(col) for col in results_df.columns]
                        results_df['Attrition_Prediction'] = preds
                        results_df['Attrition_Prediction_Label'] = results_df['Attrition_Prediction'].map({1: 'Yes', 0: 'No'})
                        
                        if pred_probs is not None:
                            results_df['Attrition_Probability'] = pred_probs.round(3)
                            
                            # Add risk level
                            def get_risk_level(prob):
                                if prob < 0.3:
                                    return "Low"
                                elif prob < 0.7:
                                    return "Medium"
                                else:
                                    return "High"
                            
                            results_df['Risk_Level'] = results_df['Attrition_Probability'].apply(get_risk_level)
                        
                        st.subheader(f"Batch Predictions using {pred_model_name}")
                        st.dataframe(ensure_string_columns(results_df))
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv,
                            file_name='attrition_predictions.csv',
                            mime='text/csv',
                        )
                        
                        # Visualization of predictions
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 8))
                            prediction_counts = results_df['Attrition_Prediction_Label'].value_counts()
                            plt.pie(
                                prediction_counts, 
                                labels=prediction_counts.index, 
                                autopct='%1.1f%%', 
                                colors=['lightblue', 'salmon'] if len(prediction_counts) > 1 else ['lightblue'],
                                explode=[0.05, 0] if len(prediction_counts) > 1 else [0]
                            )
                            plt.title('Predicted Attrition Distribution')
                            plt.axis('equal')
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        if pred_probs is not None:
                            with col2:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                plt.hist(pred_probs, bins=20, alpha=0.7, color='skyblue')
                                plt.title('Distribution of Attrition Probabilities')
                                plt.xlabel('Probability of Attrition')
                                plt.ylabel('Number of Employees')
                                plt.grid(alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        # Show high risk employees
                        high_risk = results_df[results_df.get('Risk_Level', 'None') == 'High']
                        if 'Risk_Level' in results_df.columns and not high_risk.empty:
                            st.subheader("High Risk Employees")
                            st.dataframe(ensure_string_columns(high_risk))
                            
                            # Show percentage of high-risk employees
                            risk_counts = results_df['Risk_Level'].value_counts(normalize=True) * 100
                            st.write(f"Percentage of high-risk employees: {risk_counts.get('High', 0):.1f}%")
                            
                            # Plot risk level distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.countplot(x='Risk_Level', data=results_df, order=['Low', 'Medium', 'High'], palette=['green', 'orange', 'red'])
                            plt.title("Distribution of Attrition Risk Levels")
                            plt.xlabel("Risk Level")
                            plt.ylabel("Count")
                            
                            # Add percentage labels on bars
                            for p in ax.patches:
                                height = p.get_height()
                                ax.annotate(f'{height/len(results_df):.1%}', 
                                            (p.get_x() + p.get_width() / 2., height + 0.1),
                                            ha = 'center')
                            
                            st.pyplot(fig)
                            plt.close(fig)
                            
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        st.write("Please check that your data format matches the required format.")
                        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

# Import required for accuracy calculation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    main()