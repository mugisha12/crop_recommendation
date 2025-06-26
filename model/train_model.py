import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os # For checking file existence
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load and Clean Data (reusing and adapting previous function) ---
def load_and_clean_data(file_path):
    """
    Loads the dataset, checks for missing values, handles duplicates,
    and displays basic descriptive statistics.

    Args:
        file_path (str): The path to the CSV dataset file.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    print(f"--- Loading data from: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'. Please ensure the CSV file is in the same directory.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

    print("\n--- Initial DataFrame Info ---")
    df.info()

    print("\n--- First 5 rows of the dataset ---")
    print(df.head())

    print("\n--- Checking for missing values ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print("Missing values found. Consider strategies like imputation or removal if necessary.")
        # For simplicity, we'll proceed assuming no critical missing values for this project.

    print("\n--- Checking for duplicate rows ---")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_duplicates = df.shape[0]
    if initial_rows - rows_after_duplicates > 0:
        print(f"Removed {initial_rows - rows_after_duplicates} duplicate rows.")
    else:
        print("No duplicate rows found.")

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Data Cleaning Complete ---")
    return df

# --- Step 2: Main execution block ---
if __name__ == "__main__":
    csv_file_path = 'Crop_recommendation.csv'
    df = load_and_clean_data(csv_file_path)

    if df is None:
        print("Exiting due to data loading/cleaning issues.")
    else:
        print("\n--- Data Preparation for Modeling ---")
        # Separate features (X) and target (y)
        # Assuming 'label' is the target column and all other columns are features.
        if 'label' not in df.columns:
            print("Error: 'label' column not found in the dataset. Please adjust the target column name if it's different.")
            exit() # Exit if the target column is missing

        X = df.drop('label', axis=1) # Features
        y = df['label']             # Target variable

        # Split data into training and testing sets (80% train, 20% test)
        # random_state ensures reproducibility of the split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training features shape: {X_train.shape}")
        print(f"Testing features shape: {X_test.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Testing target shape: {y_test.shape}")

        print("\n--- Model Training and Evaluation ---")
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42), # Increased max_iter for convergence
            'Support Vector Machine': SVC(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }

        best_model_name = None
        best_accuracy = 0.0
        best_trained_model = None

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_trained_model = model


        # --- Visualizations with Pie Chart, Bar Chart, and Histogram ---

        # Pie Chart: Distribution of target labels
        label_counts = df['label'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Crop Label Distribution (Pie Chart)")
        plt.axis('equal')  # Ensures pie is drawn as a circle.
        plt.tight_layout()
        plt.show()


        # Histogram: Distribution of a sample numeric feature (e.g., 'N')
        # Replace 'N' with any numeric column from your dataset as needed
        if 'N' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['N'], bins=20, color='skyblue', edgecolor='black')
            plt.title("Histogram of Nitrogen Levels (N)")
            plt.xlabel("N Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'N' not found for histogram. Replace with another numeric column.")


        if best_model_name:
            print(f"\n--- Best Model Found: {best_model_name} with Accuracy: {best_accuracy:.4f} ---")

            # --- Step 3: Save the Best Model ---
            model_filename = 'best_crop_recommender_model.joblib'
            try:
                joblib.dump(best_trained_model, model_filename)
                print(f"Best model saved successfully as '{model_filename}'")
            except Exception as e:
                print(f"Error saving the model: {e}")
        else:
            print("\nNo best model determined. Something might have gone wrong with training.")

    print("\n--- ML Pipeline Complete ---")