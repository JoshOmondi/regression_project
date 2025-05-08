# src/main.py

import os
from data_loader import load_data
from preprocess import preprocess_data
from train import train_model
import joblib

def main():
    """Main pipeline to load, preprocess, train, evaluate, and save the model."""
    
    # Step 1: Load data
    print("Loading data...")
    file_path = 'data/cleaned_housing.csv'
    df = load_data(file_path)
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df, 'median_house_value')
    
    # Step 3: Train the model
    print("Training the model...")
    model = train_model(X_train_scaled, y_train)
    
    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test_scaled, y_test)
    
    # Step 5: Save the model and scaler
    print("Saving the model and scaler...")
    save_model(model, 'models/random_forest.pkl')
    save_model(scaler, 'models/scaler.pkl')

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate the trained model and print evaluation metrics."""
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')

def save_model(model, file_path):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

if __name__ == "__main__":
    main()
