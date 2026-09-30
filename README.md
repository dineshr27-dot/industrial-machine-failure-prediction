# Industrial Machine Failure Prediction

## 1. Project Overview

Industrial machines can experience failures because of different operating conditions such as temperature, rotational speed, torque, and tool wear.

This project uses Machine Learning to predict whether an industrial machine is likely to fail based on its operating conditions.

The project is implemented as an end-to-end Machine Learning application with a Streamlit interface.

---

## 2. Problem Statement

The objective of this project is to develop a Machine Learning classification model that predicts machine failure from industrial machine operating parameters.

The prediction is binary:

- 0 → No Machine Failure
- 1 → Machine Failure

---

## 3. Objectives

The main objectives of this project are:

- Collect and prepare an industrial predictive-maintenance dataset.
- Perform data cleaning and exploratory data analysis.
- Identify useful machine operating features.
- Train multiple Machine Learning classification models.
- Compare model performance using multiple evaluation metrics.
- Select a final model.
- Develop a Streamlit application for real-time prediction.

---

## 4. Dataset

Dataset used:

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains 10,000 machine records and 14 original columns.

Important features include:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Target variable:

- Machine failure

The target contains two classes:

- 0 → No Failure
- 1 → Failure

The dataset is imbalanced, with considerably fewer failure records than non-failure records.

---

## 5. Features Used

The final Machine Learning model uses the following features:

1. Type
2. Air temperature [K]
3. Process temperature [K]
4. Rotational speed [rpm]
5. Torque [Nm]
6. Tool wear [min]

The following identifier columns were removed:

- UDI
- Product ID

Failure-mode indicator columns were not used as predictive inputs in order to avoid information leakage and keep the prediction based on machine operating conditions.

---

## 6. Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Model Saving

- Joblib

### Web Application

- Streamlit

### Development Environment

- Visual Studio Code

---

## 7. Machine Learning Workflow

The project follows this workflow:

Dataset
↓
Data Inspection
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Selection
↓
Train/Test Split
↓
Data Preprocessing
↓
Model Training
↓
Model Evaluation
↓
Model Comparison
↓
Final Model
↓
Streamlit Application

---

## 8. Data Preprocessing

The following preprocessing steps were performed:

### Duplicate Removal

Duplicate records were checked and removed.

### Identifier Removal

UDI and Product ID were removed because they are identifiers rather than useful predictive features.

### Numerical Feature Scaling

StandardScaler was used for numerical features.

### Categorical Encoding

OneHotEncoder was used to convert the machine Type feature into a numerical representation.

---

## 9. Train/Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

A stratified split was used because the target variable is imbalanced.

Final split:

- Training records: 8,000
- Testing records: 2,000

---

## 10. Machine Learning Models

Three classification algorithms were trained and compared:

### Logistic Regression

Used as a baseline classification model.

### Decision Tree

Used to learn decision rules from machine operating conditions.

### K-Nearest Neighbors (KNN)

Used to classify a machine based on similar observations in the training data.

---

## 11. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Because the dataset is imbalanced, multiple evaluation metrics were considered rather than relying only on accuracy.

---

## 12. Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 24.19% | 90.70% |
| Decision Tree | 97.90% | 73.21% | 60.29% | 66.13% | 79.76% |
| KNN | 97.40% | 83.33% | 29.41% | 43.48% | 82.91% |

Based on the F1-score criterion used in the training pipeline, the Decision Tree was selected as the final model.

The trained model is saved as:

`models/best_model.pkl`

---

## 13. Streamlit Application

A Streamlit web application was developed for machine failure prediction.

The user can enter:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The application then predicts:

- No Machine Failure
- Machine Failure

It also displays the estimated failure probability.

---

## 14. Project Structure

```text
industrial-machine-failure/
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── knn_pipeline.pkl
│   └── model_comparison.csv
│
├── plots/
│   ├── machine_failure_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── machine_type_distribution.png
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_knn.png
│   └── roc_curves.png
│
├── notebooks/
│   └── industrial_machine_failure.ipynb
│
├── train.py
├── app.py
├── requirements.txt
└── README.md