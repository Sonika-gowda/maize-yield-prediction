# 🌽 Maize Yield Prediction Under Rainfall Variability

## Project Overview

This project predicts maize crop yield under different rainfall conditions using Machine Learning.

A Decision Tree Regressor model is trained on historical crop yield data and environmental factors such as rainfall, temperature, pesticide usage, year, and country.

The project also includes SHAP Explainability to understand how different features influence yield prediction.

---

## Objectives

* Predict maize yield using environmental factors.
* Analyze the impact of rainfall variability on crop production.
* Compare predicted yield with average yield.
* Provide recommendations based on rainfall and temperature conditions.
* Explain predictions using SHAP feature attribution.

---

## Dataset

Dataset Source:

https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

Dataset contains:

* Country (Area)
* Crop Type
* Year
* Average Rainfall
* Average Temperature
* Pesticide Usage
* Crop Yield

For this project, only **Maize** records are used.

---

## Machine Learning Algorithm

* Decision Tree Regressor
* GridSearchCV for Hyperparameter Tuning
* SHAP Explainability

---

## Features

### Dataset Overview

* Yield Distribution
* Rainfall Distribution
* Top Yield Producing Countries

### Prediction Module

* Yield Prediction
* Prediction Interpretation
* Recommendations

### Analysis Module

* Rainfall vs Yield
* Temperature vs Yield

### Explainability

* Feature Importance
* SHAP Summary Plot

---

## Model Performance

* Best Max Depth: 15
* MAE: 3061.01
* RMSE: 6834.06
* R² Score: 0.9364

The model explains approximately 93.64% of yield variation.

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* SHAP
* Streamlit

---

## Run the Project

Install dependencies:

pip install -r requirements.txt

Run Streamlit app:

python -m streamlit run app.py

---

## Author

Sonika
