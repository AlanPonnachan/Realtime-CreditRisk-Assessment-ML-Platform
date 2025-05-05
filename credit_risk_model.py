

# --- Train and Save Scikit-learn Equivalent ---
# (Run this in Colab or a separate Python script after generating the CSV)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib # For saving the pipeline

# --- Configuration ---
RANDOM_SEED = 42
BATCH_CSV_PATH = f"/content/simulated_batch_loans_100000.csv" # Use the same CSV generated before
SKLEARN_MODEL_SAVE_PATH = "./sklearn_lr_credit_risk_pipeline.joblib"

# Load data using Pandas
df = pd.read_csv(BATCH_CSV_PATH)
print(f"Loaded {len(df)} records from {BATCH_CSV_PATH}")

# Define Features (X) and Target (y)
# Exclude Loan_ID and Application_Timestamp for modeling
X = df.drop(['Loan_ID', 'Default_Flag', 'Application_Timestamp'], axis=1)
y = df['Default_Flag']

# Identify feature types (ensure consistency with Spark pipeline)
categorical_features = ['Monitoring_Intensity']
numerical_features = ['Credit_Score', 'DTI_Ratio', 'Loan_Amount', 'Employment_Length_Years',
                      'Policy_Adherence_Score', 'Analysis_Depth_Score', 'Collateral_Secured_Flag']

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Create preprocessing steps for Scikit-learn Pipeline
# Note: handle_unknown='ignore' is important for new data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features) # drop='first' if used in Spark too
    ],
    remainder='passthrough' # Keep other columns if any (shouldn't be any here)
)

# Create the full pipeline with Logistic Regression
# Use class_weight='balanced' for imbalanced data
sklearn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=1000))
])

# Train the pipeline
print("Training Scikit-learn pipeline...")
sklearn_pipeline.fit(X_train, y_train)
print("Training complete.")

# Evaluate (optional, but good practice)
print("\nEvaluating Scikit-learn model...")
y_pred_proba = sklearn_pipeline.predict_proba(X_test)[:, 1]
y_pred_class = sklearn_pipeline.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred_class)
print(f"Scikit-learn Test AUC: {auc:.4f}")
print(f"Scikit-learn Test Accuracy: {acc:.4f}")

# Save the fitted pipeline
print(f"\nSaving Scikit-learn pipeline to: {SKLEARN_MODEL_SAVE_PATH}")
joblib.dump(sklearn_pipeline, SKLEARN_MODEL_SAVE_PATH)
print("Pipeline saved successfully.")