
# Employee Attrition Prediction App

This project is a machine learning-based web application that predicts the likelihood of employee attrition using a trained model and provides insightful visualizations to understand the contributing factors. Built using Streamlit, it enables HR professionals and analysts to interactively analyze employee data and get real-time attrition predictions.

## 📁 Project Structure

```
.
├── employeeAttritionPred.py         # Contains the machine learning model class
├── attritionPredApp.py             # Main Streamlit app for predictions
├── employeeAttritionVisualizations.py # Visualization tools for EDA and feature insights
└── README.md                       # Project documentation
```

## 🚀 Features

- **Interactive UI with Streamlit**  
  User-friendly interface for inputting employee data and receiving prediction results.

- **Machine Learning Model**  
  A class-based predictive model trained to detect employee attrition likelihood.

- **Visual Analytics**  
  Displays pie charts, bar graphs, and correlation heatmaps to understand feature impacts.

- **Explainability**  
  Highlights the most important features affecting model predictions.

## 🧠 Model Overview

The `EmployeeAttritionModel` in `employeeAttritionPred.py` handles:
- Data preprocessing
- Feature encoding
- Model training using multiple algorithms
- Hyperparameter tuning with GridSearchCV
- SMOTE for class imbalance
- Feature importance and SHAP value extraction
- Prediction pipeline

## 📊 Visualization Module

The `employeeAttritionVisualizations.py` provides:
- **Attrition Distribution**
- **Numerical and Categorical Feature Distributions**
- **Correlation Heatmap**
- **Model Comparison Metrics**
- **ROC & PR Curves**
- **SHAP Value Visualizations**
- **Permutation Importance**
- **Risk Heatmaps and Department Profiling**

## 🖥️ Streamlit App Usage

The main app (`attritionPredApp.py`) does the following:
1. Loads the model and visualization utilities.
2. Displays EDA and model insights.
3. Allows model training and evaluation.
4. Supports batch predictions via CSV upload.
5. Visualizes risk distribution and top influential features.

## ⚙️ How to Run

```bash
pip install -r requirements.txt
streamlit run attritionPredApp.py
```

## ✅ Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- xgboost
- shap
- imbalanced-learn

## 📌 Notes

- Ensure the dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) is placed in the correct directory.
- The app uses SHAP for model explainability, with fallbacks for non-compatible models.
- Visualization and interpretability features require trained models.

