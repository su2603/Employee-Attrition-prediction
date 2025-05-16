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
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Model Training & Evaluation", "Batch Prediction"])
    
    if page == "Model Training & Evaluation":
        model_training_page(model)
    elif page == "Batch Prediction":
        batch_prediction_page(model)

def model_training_page(model):
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
    
    # Progress bar for model training
    progress_bar = st.progress(0)
    model_scores = {}
    
    # Train and evaluate models
    for i, model_name in enumerate(selected_models):
        st.subheader(f"Training {model_name}...")
        metrics = model.train_model(model_name, run_tuning=run_tuning)
        
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
        
        # Show SHAP values for tree-based models
        if model_name in ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree"]:
            shap_data = model.get_shap_values(model_name)
            if shap_data is not None:
                st.write(f"SHAP summary plot for {model_name} (sample of test data):")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(
                    shap_data['shap_values'], 
                    shap_data['X_sample'], 
                    feature_names=shap_data['feature_names'], 
                    show=False
                )
                st.pyplot(fig)
                plt.clf()
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(selected_models))
    
    # Show model comparison
    if model_scores:
        st.subheader("Model Comparison")
        metrics_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={'index': 'Model'})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', x='Model', ax=ax)
        plt.title("Model Performance Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
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
        else:
            st.write("Feature importance not available for this model type")

def batch_prediction_page(model):
    st.header("Batch Prediction: Upload CSV")
    
    uploaded_file = st.file_uploader("Upload CSV with employee data for batch prediction", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_data.head())
            
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
            try:
                prediction_results = model.predict(new_data, pred_model_name)
                
                # Get predictions and probabilities
                preds = prediction_results['predictions']
                pred_probs = prediction_results['probabilities']
                
                # Display results
                results_df = pd.DataFrame(new_data)
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
                st.dataframe(results_df)
                
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
                
                if pred_probs is not None:
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        plt.hist(pred_probs, bins=20, alpha=0.7, color='skyblue')
                        plt.title('Distribution of Attrition Probabilities')
                        plt.xlabel('Probability of Attrition')
                        plt.ylabel('Number of Employees')
                        plt.grid(alpha=0.3)
                        st.pyplot(fig)
                
                # Show high risk employees
                high_risk = results_df[results_df.get('Risk_Level', 'None') == 'High']
                if 'Risk_Level' in results_df.columns and not high_risk.empty:
                    st.subheader("High Risk Employees")
                    st.dataframe(high_risk)
                    
                    # Show percentage of high-risk employees
                    risk_counts = results_df['Risk_Level'].value_counts(normalize=True) * 100
                    st.write(f"Percentage of high-risk employees: {risk_counts.get('High', 0):.1f}%")
                    
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.write("Please check that your data format matches the required format.")
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

if __name__ == "__main__":
    main()