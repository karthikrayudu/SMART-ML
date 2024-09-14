import pandas as pd

# Correct file path with raw string notation
file_path = r"C:\sd card\ip projects\smartml\smartml\dataset\hugedata.csv"

try:
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check for missing values
    missing_values = data.isna().sum()
    print("Missing values in each column:\n", missing_values)

    # Display the structure of the data
    print("\nData structure:\n", data.info())

    # Display the summary of the data
    print("\nSummary of the data:\n", data.describe())

    # Convert the 'Class' column to categorical type
    data['Class'] = data['Class'].astype('category')
    print("\nUpdated data types:\n", data.dtypes)
    # Display the summary of the data
    print("\nSummary of the data:\n", data.describe())

except FileNotFoundError:
    print(f"File not found: {file_path}")
