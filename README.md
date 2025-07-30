# 🌍 Global Terrorism Database Analysis

A complete end-to-end data science project analyzing global terrorism trends using machine learning and data visualization techniques.

## 📁 Project Overview

This project explores the [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd/) to understand patterns, trends, and insights from terrorist incidents worldwide. The analysis covers data preprocessing, EDA, feature engineering, and the implementation of various machine learning models to classify or predict the type of attack.

## 🧠 Objectives

- Clean and preprocess a real-world terrorism dataset.
- Perform exploratory data analysis (EDA) to uncover insights.
- Build classification models to predict the attack type or other relevant outcomes.
- Evaluate model performance using metrics like Accuracy, Precision, and Recall.
- Tune hyperparameters to improve model results.
- Compare multiple ML models including:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - XGBoost (with handling for multiclass targets)
  - Neural Network (optional/experimental)

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` for data wrangling and visualization
  - `scikit-learn` for machine learning models and metrics
  - `xgboost` for gradient boosting model
  - `tensorflow` or `keras` for optional neural networks
- **IDE**: Jupyter Notebook / VS Code

## 📊 Exploratory Data Analysis

- Year-wise and country-wise trend analysis
- Heatmaps for attack types, regions, and targets
- Visualizations to reveal hotspots and patterns
- Handling of missing data and feature encoding

## 🤖 Model Evaluation

| Model              | Accuracy | Recall | Precision |
|-------------------|----------|--------|-----------|
| Decision Tree      | 0.7897  | 0.7897  |  0.7924  |
| Random Forest      | 0.8309  | 0.8309  |  0.8221  |
| Logistic Regression| 0.6080  | 0.6080  | 0.4726   |

## ⚙️ Future Improvements

- Advanced hyperparameter tuning using GridSearchCV
- Model export using pickle or joblib


## 📂 Files Included

- `GlobalTerrorismAnalysis.ipynb` – Jupyter notebook with full analysis
- `metrics_dataframe.csv` – Model performance summary
- `README.md` – This file
- `visuals/` – Optional folder for graphs and plots

## 🧠 Learnings

- Real-world multiclass classification
- Handling large datasets
- Comparing ML models on real data
- Understanding the geopolitical impact of terrorism data

## 🙋‍♂️ Author

**Shehzad Khan**  
Feel free to connect or collaborate!

---

> ⚠️ *Disclaimer: This project is for educational purposes. The GTD data is publicly available, and all interpretations are purely academic.*


