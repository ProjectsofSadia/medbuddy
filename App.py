import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

os.makedirs("models", exist_ok=True)
try:
    data_path = r"C:\Users\Asus\Downloads\MedBuddy\processed_dataset.csv"
    if not os.path.exists(data_path):

        data_path = r"C:\Users\Asus\Downloads\MedBuddy\dataset.csv"
        print("Processed dataset not found, using original dataset...")
    
    symptoms_data = pd.read_csv(data_path)
    print("Data loaded successfully!")
    print(f"Data shape: {symptoms_data.shape}")
    print("First few rows:")
    print(symptoms_data.head())
    

    if "Disease" in symptoms_data.columns:

        diseases = symptoms_data["Disease"].unique()
        with open("models/disease_classes.pkl", "wb") as f:
            pickle.dump(diseases, f)

        X = symptoms_data.drop(columns=["Disease"])
        y = symptoms_data["Disease"]

        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Found non-numeric columns: {non_numeric_cols}")

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in non_numeric_cols:
                X[col] = le.fit_transform(X[col])
            
            print("Converted non-numeric columns to numeric.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        
        model_path = "models/symptom_checker.pkl"
        pickle.dump(model, open(model_path, "wb"))
        print(f"Model saved to {model_path}")

        with open("models/feature_names.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)
        
        print("Training complete!")
    else:
        print("Error: 'Disease' column not found in the dataset")
        
except Exception as e:
    print(f"Error during model training: {str(e)}")
