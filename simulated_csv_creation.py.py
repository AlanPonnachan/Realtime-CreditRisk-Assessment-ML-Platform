

import os
import findspark
findspark.init() # Important to find Spark installation

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import random
import pandas as pd # Still useful for initial data generation/manipulation
import numpy as np
import time
import json
from queue import Queue # To simulate Kafka queue

# Create a SparkSession (adjust config for Colab resources if needed)
spark = SparkSession.builder \
    .appName("RealWorldCreditRiskSimulation") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

sc = spark.sparkContext # Get SparkContext

print(f"SparkSession created. Spark version: {spark.version}")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Configuration for Larger Data ---
NUM_LOANS_BATCH = 100000 # Generate 100k for batch training 
DEFAULT_RATE_BATCH = 0.07 # Target approximate default rate
BATCH_DATA_PATH = f"./simulated_batch_loans_{NUM_LOANS_BATCH}.csv"

print(f"Starting large-scale data generation for {NUM_LOANS_BATCH} loans...")

def generate_loan_data(num_loans, default_rate_target):
    """Generates a pandas DataFrame of loan data."""
    # --- Generate Base Borrower Features ---
    credit_scores = np.random.normal(loc=650, scale=80, size=num_loans).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)

    dti_ratios = np.random.beta(a=2, b=5, size=num_loans) * 0.6
    dti_ratios = np.clip(dti_ratios, 0.05, 0.6)

    loan_amounts = np.random.lognormal(mean=np.log(15000), sigma=0.8, size=num_loans)
    loan_amounts = np.clip(loan_amounts, 1000, 100000).round(0)

    employment_length_years = np.random.randint(0, 25, size=num_loans)

    # --- Generate Simulated CRM Practice Features ---
    policy_adherence_score = np.random.randint(1, 6, size=num_loans)
    monitoring_intensity = np.random.choice(['Low', 'Medium', 'High'], size=num_loans, p=[0.4, 0.4, 0.2])
    analysis_depth_score = np.random.randint(1, 6, size=num_loans)
    collateral_secured_flag = np.random.choice([0, 1], size=num_loans, p=[0.7, 0.3])

    # --- Simulate Default Probability ---
    base_log_odds = np.log(default_rate_target / (1 - default_rate_target)) - 4.5
    log_odds = base_log_odds \
               - 0.015 * (credit_scores - 650) \
               + 4.0 * dti_ratios \
               + 0.00001 * (loan_amounts - 15000) \
               - 0.05 * employment_length_years \
               - 0.30 * (policy_adherence_score - 3) \
               - 0.15 * (analysis_depth_score - 3) \
               - 0.50 * collateral_secured_flag \
               + 0.10 * (monitoring_intensity == 'Low') \
               - 0.05 * (monitoring_intensity == 'High')
    log_odds += np.random.normal(loc=0, scale=1.0, size=num_loans)
    probabilities = 1 / (1 + np.exp(-log_odds))
    default_flag = (np.random.uniform(0, 1, size=num_loans) < probabilities).astype(int)

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Loan_ID': range(1, num_loans + 1),
        'Credit_Score': credit_scores,
        'DTI_Ratio': dti_ratios.round(4),
        'Loan_Amount': loan_amounts,
        'Employment_Length_Years': employment_length_years,
        'Policy_Adherence_Score': policy_adherence_score,
        'Monitoring_Intensity': monitoring_intensity,
        'Analysis_Depth_Score': analysis_depth_score,
        'Collateral_Secured_Flag': collateral_secured_flag,
        'Default_Flag': default_flag # Target Variable
    })
    # Add a timestamp for potential streaming use cases later
    df['Application_Timestamp'] = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(1, 365*24*60, size=num_loans), unit='m')

    return df

# Generate and save the large batch dataset
batch_df_pd = generate_loan_data(NUM_LOANS_BATCH, DEFAULT_RATE_BATCH)
batch_df_pd.to_csv(BATCH_DATA_PATH, index=False, header=True)

print(f"\nGenerated {len(batch_df_pd)} batch records.")
print(f"Actual generated default rate: {batch_df_pd['Default_Flag'].mean():.4f}")
print(f"Batch data saved to: {BATCH_DATA_PATH}")
print("\nBatch Data Head (Pandas):")
print(batch_df_pd.head())

