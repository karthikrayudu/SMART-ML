import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic dataset
n_samples = 100
n_features = 30  # One less than the total columns to account for the target

# Generate random features
X = np.random.randn(n_samples, n_features)

# Generate binary target with 70% accuracy
y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Combine features and target into a DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
data['target'] = y

# Save to CSV
data.to_csv('C:\sd card\ip projects\CreditCard xgboost\CreditCard xgboost\dataset\sample_data_100_columns.csv', index=False)
