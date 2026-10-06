# 💼 Employee Salary Analytics

An end-to-end Machine Learning project that predicts an employee's salary based on demographic, educational, and professional information. The project also provides salary benchmarking, what-if analysis, and interactive salary analytics through a Streamlit dashboard.

## 📌 Project Overview

Salary prediction is a regression problem where the goal is to estimate an employee's salary from available information such as:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience

Instead of only returning a predicted salary, this project provides additional context through salary benchmarks, experience-based what-if analysis, salary trends, job-title comparisons, and model feature importance.

## 🎯 Objectives

- Clean and prepare salary data for machine learning.
- Perform exploratory data analysis to understand salary patterns.
- Compare multiple regression algorithms.
- Build a reliable preprocessing and prediction pipeline.
- Select and evaluate a suitable regression model.
- Save the trained pipeline for reuse.
- Develop an interactive Streamlit application.
- Provide salary benchmarking and analytical insights alongside predictions.

## 🗂️ Project Structure

```text
Employee-Salary-Intelligence/
│
├── data/
│   └── Salary Data.csv
│
├── notebooks/
│   └── Salary_Prediction.ipynb
│
├── models/
│
├── app.py
├── salary_prediction_pipeline.pkl
├── feature_importance.csv
├── grouped_feature_importance.csv
└── requirements.txt
```

> File and folder names may vary slightly depending on the local project setup.

## 🔄 Machine Learning Workflow

```text
Salary Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Train / Test Split
      ↓
Feature Preprocessing
      ↓
Model Comparison
      ↓
Random Forest Selection
      ↓
Model Evaluation
      ↓
Pipeline Serialization
      ↓
Streamlit Application
```

## 🧹 Data Cleaning

The dataset was inspected for missing values, duplicate records, and suspicious salary values.

The following cleaning steps were applied:

- Removed completely empty rows.
- Removed duplicate rows.
- Removed rows containing remaining missing values.
- Removed salary records below $10,000 as suspicious/outlier records for this project.

After cleaning, the resulting dataset was used for analysis and model development.

## 📊 Exploratory Data Analysis

Several visualizations were used to understand the dataset, including:

- Salary distribution
- Salary vs. years of experience
- Salary vs. education level
- Correlation analysis
- Average salary by job title

The analysis showed a strong relationship between salary and experience, while age and experience were also strongly correlated.

## ⚙️ Feature Preprocessing

The project uses a `ColumnTransformer` to apply different preprocessing techniques to numerical and categorical features.

### Numerical Features

- Age
- Years of Experience

These features are standardized using `StandardScaler`.

### Categorical Features

- Gender
- Education Level
- Job Title

These features are converted using `OneHotEncoder`.

`handle_unknown="ignore"` is used so that the application can handle a categorical value that was not present during model training.

All preprocessing and model steps are combined into a single Scikit-learn `Pipeline`.

## 🤖 Models Compared

The project compares several regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors
- Support Vector Regression
- XGBoost

The final model used in the application is **Random Forest Regressor**.

It was selected considering its predictive performance, robustness, and interpretability for this project.

## 🌲 Final Model

The final model is a Random Forest Regressor with:

```text
n_estimators = 300
random_state = 42
n_jobs = -1
```

The complete preprocessing + model pipeline is saved as:

```text
salary_prediction_pipeline.pkl
```

This allows the Streamlit application to directly load the trained pipeline and make predictions without repeating the training process.

## 📈 Model Performance

On the current held-out test set, the final Random Forest model achieved approximately:

| Metric | Value |
|---|---:|
| R² | 0.885 |
| MAE | $10,562 |
| RMSE | $15,222 |

### Metric Interpretation

- **R²:** Measures how much of the variance in salary is explained by the model.
- **MAE:** Represents the average absolute difference between predicted and actual salary.
- **RMSE:** Measures prediction error while giving more weight to larger errors.

The R² score of approximately 0.885 means the model explains about 88.5% of the salary variance on the held-out test set.

## 🖥️ Streamlit Dashboard

The application provides three main sections.

### 💰 Salary Prediction

Users can enter:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience

The application then predicts the expected salary using the saved Random Forest pipeline.

### 📊 Salary Benchmark

The predicted salary is compared with:

- Overall dataset median
- Average salary for the selected job title
- Estimated salary percentile

This provides context around the prediction rather than showing an isolated number.

### 🔮 What-If Analysis

Users can change years of experience and observe how the predicted salary changes while keeping the other employee attributes fixed.

### 📈 Salary Analytics

The dashboard provides:

- Salary distribution
- Experience vs. salary analysis
- Average salary by education level
- Top 15 job titles by average salary

### 🧠 Model Insights

The application displays grouped feature importance to show which original input features contribute most to the Random Forest predictions.

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Employee-Salary-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Plotly**
- **Streamlit**
- **Joblib**
- **Jupyter Notebook**

## ⚠️ Limitations

- The dataset is relatively small.
- Job titles have a highly uneven distribution, with many titles having few observations.
- Salary is influenced by many real-world factors that are not included in the dataset.
- The model should be treated as an estimation tool rather than a source of actual compensation decisions.
- The salary benchmark represents the available dataset and should not be interpreted as a complete representation of the real-world job market.

## 🔮 Future Improvements

Possible improvements include:

- Training with a larger and more diverse dataset.
- Better handling of rare job titles.
- Hyperparameter tuning.
- Cross-validation during model selection.
- Adding location and industry as features.
- Adding a model prediction confidence/uncertainty analysis.
- Deploying the application online.

## 👨‍💻 Author

**Adarsh Mohanty**

B.Tech Computer Science Engineering

---

⭐ If you found this project useful, consider giving the repository a star.
