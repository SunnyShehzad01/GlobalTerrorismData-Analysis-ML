# 🌍 Global Terrorism Database Analysis

## 📁 Project Overview
Terrorism remains one of the most pressing global challenges, with incidents often going unattributed due to the absence of claims from perpetrator groups. This project leverages the Global Terrorism Database (1970–2021) — a rich dataset with over 214,000 incidents and 135 attributes — to build machine learning models that predict the terrorist group (gname) responsible for an attack, based on incident characteristics.
This project explores the [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd/) to understand patterns, trends, and insights from terrorist incidents worldwide. The analysis covers data preprocessing, EDA, feature engineering, and the implementation of various machine learning models to classify or predict the type of attack.
Our primary goal is to explore the potential of data-driven methods in supporting faster investigations, better resource allocation, and enhanced counter-terrorism strategies.

## 📂 Dataset Information

The project uses the **Global Terrorism Database (GTD)**, publicly available for research.
Some important columns include:

* **`gname` (Target Variable)** → Terrorist group name responsible for the attack.
* **`country_txt`** → Country where the incident took place.
* **`attacktype1_txt`** → Type of attack (e.g., Bombing, Armed Assault, Assassination).
* **`targtype1_txt`** → Primary target type (e.g., Military, Police, Private Citizens, Business).
* **`weaptype1_txt`** → Type of weapon used.
* **`region_txt`** → Geographical region of the incident.
* **`nkill`** → Number of people killed.
* **`nwound`** → Number of people wounded.
* **Other features** such as year, month, success of attack, suicide indicator, etc.

---
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
  - `xgboost` & `adaboost` for xtreme-gradient & Adaptive boosting model
- **IDE**: Jupyter Notebook / VS Code

---

## 🔍 Exploratory Data Analysis (EDA)

We performed extensive **data exploration and visualization** to identify key insights, including:

* Trends of terrorist attacks over time.
* Most targeted countries, regions, and sectors.
* Common attack types and weapon types.
* Correlation of features with the target variable (`gname`).
* Handling of **missing values**, **class imbalance**, and **noisy data**.

Color maps like `viridis`, `plasma`, and `RdBu` were used for clear and impressive visualizations.

---

## ⚙️ Methodology

### 1. **Data Preprocessing**

* Cleaned missing and inconsistent data.
* Encoded categorical variables.
* Normalized numerical values.
* Addressed **class imbalance** with appropriate techniques.

### 2. **Model Building**

Applied multiple machine learning algorithms:

* Logistic Regression
* Decision Tree
* Random Forest ✅
* AdaBoost
* XGBoost

### 3. **Evaluation & Tuning**

* Data split using **Train-Test Split**.
* Model evaluation using **Accuracy, Precision, Recall, and F1-score**.
* Achieved **82% accuracy** with **Random Forest**, which performed best.
* Performed **GridSearchCV** for hyperparameter tuning.
* Applied **K-Fold Cross Validation** for robust evaluation.

---

## 📊 Results

* **Best Model:** Random Forest Classifier
* **Accuracy:** **82%**
* Hyperparameter tuning and validation improved consistency of results.

---

## 🚀 Future Work
* Advanced hyperparameter tuning using GridSearchCV
* Experiment with **deep learning models (LSTMs/Transformers)** for text-based features.
* Implement **real-time prediction pipelines**.
* Explore **explainable AI (XAI)** methods for better interpretability.

---

## 📂 Files Included

- `GTD Analysis.ipynb` – Jupyter notebook with full analysis
- `README.md` – This file

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


