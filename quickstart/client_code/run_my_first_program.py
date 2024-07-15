# nada_program.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 1: Load Data
def load_data():
    # Placeholder for loading data
    # For example purposes, create a simple dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    X = df.iloc[:, :-1].values  # Features
    y = df['target'].values      # Target variable
    return X, y

# Step 3: Split Data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Main function to execute the program
def main():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
