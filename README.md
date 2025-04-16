# House Price Prediction Using Regression Models

This project predicts house prices based on 79 property features using multiple regression models and ensemble techniques. Built for the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), it demonstrates a complete machine learning pipeline from preprocessing to model stacking.

## Objective
Predict the final sale price of homes in Ames, Iowa using various property characteristics (e.g., area, quality, age, amenities) with minimum root mean squared log error (RMSLE).

##  Approach Overview
- **Data Cleaning:** Handled missing values using median imputation and encoded categorical variables
- **Feature Engineering:** Created new features like `TotalSF`, `HasGarage`, and log-transformed skewed numerical columns
- **EDA:** Visualized target distribution, feature correlations, and missing data
- **Model Training:** Trained and compared Linear Regression, Random Forest, XGBoost, and LightGBM
- **Ensembling:** Built a stacked model using RidgeCV to combine base models
- **Evaluation Metric:** Root Mean Squared Log Error (RMSLE)

##  Results
- **Final RMSLE:** `0.13158` on Kaggle public leaderboard
- **Leaderboard Rank:** 1323 / 4800+ (Top ~28%)

##  Models Used
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (tuned using GridSearchCV)
- LightGBM Regressor
- Ridge Regression (for stacking ensemble)

##  File Structure
├── house_price_prediction.ipynb # Main notebook ├── submission.csv # Sample output file ├── README.md # This file └── requirements.txt # Dependencies


##  Visualizations Included
- Feature Importance (XGBoost)
- RMSE Comparison Across Models
- Distribution plots of target variable
- Heatmap of feature correlations




