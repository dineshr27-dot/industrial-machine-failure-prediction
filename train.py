# ============================================================
# INDUSTRIAL MACHINE FAILURE PREDICTION
# Problem 21 - Learn Depth Capstone
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("INDUSTRIAL MACHINE FAILURE PREDICTION")
print("=" * 60)

df = pd.read_csv("data/ai4i2020.csv")

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATA INSPECTION")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# 5. REMOVE IDENTIFIER COLUMNS
# ============================================================

# UDI and Product ID identify individual records/machines.
# They are not useful predictive features.

df = df.drop(
    columns=["UDI", "Product ID"],
    errors="ignore"
)

print("\nColumns after removing identifiers:")
print(df.columns.tolist())


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nMachine Failure counts:")

print(
    df["Machine failure"].value_counts()
)

print("\nMachine Failure percentage:")

print(
    df["Machine failure"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 7. MACHINE FAILURE DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")

plt.xlabel(
    "Machine Failure (0 = No Failure, 1 = Failure)"
)

plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    "plots/machine_failure_distribution.png"
)

plt.show()


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(
    df.describe()
)


# ============================================================
# 9. NUMERICAL FEATURES
# ============================================================

numeric_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ============================================================
# 10. FEATURE DISTRIBUTIONS
# ============================================================

df[numeric_columns].hist(
    figsize=(12, 8),
    bins=30
)

plt.suptitle(
    "Distribution of Machine Operating Features"
)

plt.tight_layout()

plt.savefig(
    "plots/feature_distributions.png"
)

plt.show()


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

correlation = df[
    numeric_columns + ["Machine failure"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Between Machine Features and Failure"
)

plt.tight_layout()

plt.savefig(
    "plots/correlation_heatmap.png"
)

plt.show()


# ============================================================
# 12. MACHINE TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("MACHINE TYPE DISTRIBUTION")
print("=" * 60)

print(
    df["Type"].value_counts()
)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Type"
)

plt.title(
    "Machine Type Distribution"
)

plt.xlabel("Machine Type")

plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    "plots/machine_type_distribution.png"
)

plt.show()


# ============================================================
# 13. FEATURE SELECTION
# ============================================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"


X = df[features]

y = df[target]


print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

print("\nSelected Features:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)

print("\nFeature shape:", X.shape)

print("Target shape:", y.shape)


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining data:", X_train.shape)

print("Testing data:", X_test.shape)

print("\nTraining target distribution:")

print(
    y_train.value_counts()
)

print("\nTesting target distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 15. PREPROCESSING
# ============================================================

categorical_features = [
    "Type"
]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


print("\nPreprocessing pipeline created successfully!")


# ============================================================
# 16. CREATE MACHINE LEARNING MODELS
# ============================================================

print("\n" + "=" * 60)
print("CREATING MACHINE LEARNING MODELS")
print("=" * 60)


# -----------------------------
# Logistic Regression
# -----------------------------

logistic_model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# -----------------------------
# Decision Tree
# -----------------------------

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# -----------------------------
# KNN
# -----------------------------

knn_model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)


print("\nModels created:")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. KNN")


# ============================================================
# 17. TRAIN MODELS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)


print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed!")


print("\nTraining Decision Tree...")

decision_tree_model.fit(
    X_train,
    y_train
)

print("Decision Tree training completed!")


print("\nTraining KNN...")

knn_model.fit(
    X_train,
    y_train
)

print("KNN training completed!")


# ============================================================
# 18. MAKE PREDICTIONS
# ============================================================

logistic_pred = logistic_model.predict(
    X_test
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

knn_pred = knn_model.predict(
    X_test
)


# ============================================================
# 19. PREDICT PROBABILITIES
# ============================================================

logistic_prob = logistic_model.predict_proba(
    X_test
)[:, 1]

decision_tree_prob = decision_tree_model.predict_proba(
    X_test
)[:, 1]

knn_prob = knn_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 20. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_prob
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    print("\n" + "-" * 60)

    print(model_name)

    print("-" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 21. EVALUATE ALL MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


results = []


results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_pred,
        logistic_prob
    )
)


results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        decision_tree_pred,
        decision_tree_prob
    )
)


results.append(
    evaluate_model(
        "KNN",
        y_test,
        knn_pred,
        knn_prob
    )
)


# ============================================================
# 22. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# Save comparison results

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# ============================================================
# 23. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)


print("\nLogistic Regression:")

print(
    classification_report(
        y_test,
        logistic_pred,
        zero_division=0
    )
)


print("\nDecision Tree:")

print(
    classification_report(
        y_test,
        decision_tree_pred,
        zero_division=0
    )
)


print("\nKNN:")

print(
    classification_report(
        y_test,
        knn_pred,
        zero_division=0
    )
)


# ============================================================
# 24. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

cm_logistic = confusion_matrix(
    y_test,
    logistic_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix_logistic.png"
)

plt.show()


# ============================================================
# 25. CONFUSION MATRIX - DECISION TREE
# ============================================================

cm_tree = confusion_matrix(
    y_test,
    decision_tree_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Decision Tree"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix_decision_tree.png"
)

plt.show()


# ============================================================
# 26. CONFUSION MATRIX - KNN
# ============================================================

cm_knn = confusion_matrix(
    y_test,
    knn_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - KNN"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix_knn.png"
)

plt.show()


# ============================================================
# 27. ROC CURVES
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

tree_fpr, tree_tpr, _ = roc_curve(
    y_test,
    decision_tree_prob
)

knn_fpr, knn_tpr, _ = roc_curve(
    y_test,
    knn_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label="Logistic Regression"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label="Decision Tree"
)

plt.plot(
    knn_fpr,
    knn_tpr,
    label="KNN"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Machine Failure Prediction"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/roc_curves.png"
)

plt.show()


# ============================================================
# 28. SELECT MODEL BASED ON F1 SCORE
# ============================================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]


print("\n" + "=" * 60)

print(
    "MODEL WITH HIGHEST F1 SCORE:"
)

print(
    best_model_name
)

print("=" * 60)


# ============================================================
# 29. SAVE ALL TRAINED MODELS
# ============================================================

joblib.dump(
    logistic_model,
    "models/logistic_regression_pipeline.pkl"
)

joblib.dump(
    decision_tree_model,
    "models/decision_tree_pipeline.pkl"
)

joblib.dump(
    knn_model,
    "models/knn_pipeline.pkl"
)


print("\nAll models saved successfully!")


# ============================================================
# 30. SAVE THE BEST MODEL
# ============================================================

if best_model_name == "Logistic Regression":

    best_model = logistic_model

elif best_model_name == "Decision Tree":

    best_model = decision_tree_model

else:

    best_model = knn_model


joblib.dump(
    best_model,
    "models/best_model.pkl"
)


print(
    "\nBest model saved as:"
)

print(
    "models/best_model.pkl"
)


# ============================================================
# 31. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "TRAINING PIPELINE COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print("\nGenerated files:")

print("models/model_comparison.csv")

print("models/logistic_regression_pipeline.pkl")

print("models/decision_tree_pipeline.pkl")

print("models/knn_pipeline.pkl")

print("models/best_model.pkl")

print("\nGenerated plots:")

print("plots/machine_failure_distribution.png")

print("plots/feature_distributions.png")

print("plots/correlation_heatmap.png")

print("plots/machine_type_distribution.png")

print("plots/confusion_matrix_logistic.png")

print("plots/confusion_matrix_decision_tree.png")

print("plots/confusion_matrix_knn.png")

print("plots/roc_curves.png")

print("\nNext step: Build the Streamlit application.")