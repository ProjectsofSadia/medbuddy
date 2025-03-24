import pandas as pd
import streamlit as st
import pickle
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from PIL import Image

st.set_page_config(page_title="MedBuddy AI Medical Advisor", page_icon="🩺")

st.markdown("""
<style>
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stTextInput input {
        background-color: #F0F2F6;
    }
    .stMarkdown h1 {
        color: #FF4B4B;
    }
    .stMarkdown h2 {
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

le = LabelEncoder()


os.makedirs(os.path.join(os.path.expanduser("~"), "MedBuddy_models"), exist_ok=True)


if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'reminders' not in st.session_state:
    st.session_state.reminders = []

st.write(f"Data Loaded: {st.session_state.data_loaded}")
st.write(f"Model Trained: {st.session_state.model_trained}")


with st.sidebar:
    st.header("Settings")
    st.write("Configure your app here.")
    user_name = st.text_input("Enter your name")
    if user_name:
        st.write(f"Hello, {user_name}!")


st.title("🩺 MedBuddy AI Medical Advisor")
st.markdown("""
Your AI-powered healthcare assistant for symptom checking, health questions, and medication reminders.

**Note:** This is for educational purposes only and not a substitute for professional medical advice.
""")


image = Image.open(r"C:\Users\Asus\Downloads\MedBuddy\medbuddy_image.jpg")  
st.image(image, caption="MedBuddy AI", use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["Setup", "Symptom Checker", "Health Chatbot", "Medicine Reminder"])


def find_dataset():
    
    data_path = [
        r"C:\Users\Asus\Downloads\MedBuddy\dataset.csv",
        Path.cwd() / "dataset.csv",
        Path.cwd().parent / "dataset.csv",
    ]
    
    for path in data_path:
        try:
            if os.path.exists(str(path)):
                return str(path)
        except:
            continue
    
    return None

def load_data():
    try:
        dataset_path = r"C:\Users\Asus\Downloads\MedBuddy\dataset.csv"
        
        if dataset_path is None:
            st.error("Dataset file not found. Please upload a dataset file.")
            uploaded_file = st.file_uploader("Upload your dataset CSV", type=['csv'])
            if uploaded_file is not None:
                symptoms_data = pd.read_csv(uploaded_file)
                st.session_state.data_loaded = True
                return symptoms_data
            return None
        
        symptoms_data = pd.read_csv(dataset_path)
        
        if not symptoms_data.empty:
            st.session_state.data_loaded = True
            return symptoms_data
        else:
            st.error("Dataset is empty. Please check your data file.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(symptoms_data):
    """Preprocess the dataset for model training."""
    if "Disease" not in symptoms_data.columns:
        st.error("'Disease' column not found in the dataset")
        return None, None
        
    X = symptoms_data.drop(columns=["Disease"])
    y = symptoms_data["Disease"]

    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        st.info(f"Converting non-numeric columns to numeric format")

        all_symptoms = set()
        for col in non_numeric_cols:
            X[col] = X[col].fillna("")
            all_symptoms.update(X[col].unique())

        if "" in all_symptoms:
            all_symptoms.remove("")

        st.session_state.all_symptoms = sorted(list(all_symptoms))

        for col in non_numeric_cols:
            X[col] = le.fit_transform(X[col])
    
    return X, y

def train_model(symptoms_data):
    try:
        st.write("Dataset information:")
        st.write(f"Shape: {symptoms_data.shape}")
        st.write("Sample data:")
        st.write(symptoms_data.head())
        
        X, y = preprocess_data(symptoms_data)
        if X is None or y is None:
            return None, None
            
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner("Training model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            
            accuracy = model.score(X_test, y_test)
            st.success(f"Model trained with accuracy: {accuracy:.2f}")
            
            if accuracy == 1.0:
                st.warning("100% accuracy might indicate overfitting. Consider using cross-validation for a more robust evaluation.")

            
            scores = cross_val_score(model, X, y, cv=5)
            st.write(f"Cross-validation scores: {scores}")
            st.write(f"Average accuracy: {scores.mean():.2f}")

            
            model_path = "models/symptom_checker.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)

            diseases = sorted(y.unique())
            with open("models/disease_classes.pkl", "wb") as f:
                pickle.dump(diseases, f)
            
            with open("models/feature_names.pkl", "wb") as f:
                pickle.dump(list(X.columns), f)
            
            st.session_state.model_trained = True
            st.session_state.feature_names = list(X.columns)
            st.session_state.model = model
            st.session_state.diseases = diseases
            
            return model, list(X.columns)
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def load_trained_model():
    """Load a previously trained model."""
    try:
        model_path = "models/symptom_checker.pkl"
        feature_names_path = "models/feature_names.pkl"
        disease_classes_path = "models/disease_classes.pkl"
        
        if not all(os.path.exists(p) for p in [model_path, feature_names_path, disease_classes_path]):
            st.warning("Complete model files not found. Please train a new model.")
            return None, None
        
        model = joblib.load(model_path)
        feature_names = pickle.load(open(feature_names_path, "rb"))
        diseases = pickle.load(open(disease_classes_path, "rb"))
        
        if 'all_symptoms' not in st.session_state and os.path.exists("models/all_symptoms.pkl"):
            try:
                all_symptoms = pickle.load(open("models/all_symptoms.pkl", "rb"))
                st.session_state.all_symptoms = all_symptoms
            except:
                st.warning("Could not load symptoms list. Some features may be limited.")
        
        st.success("Model loaded successfully!")
        st.session_state.model_trained = True
        st.session_state.feature_names = feature_names
        st.session_state.model = model
        st.session_state.diseases = diseases
        
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_disease(symptoms, model, feature_names):
    """Predict disease based on symptoms."""
    try:
        input_vector = np.zeros(len(feature_names))
        if hasattr(st.session_state, 'all_symptoms'):
            for i, feature in enumerate(feature_names):
                for symptom in symptoms:
                    if symptom in feature:
                        input_vector[i] = 1
                        break
        else:
            input_vector = np.array([1 if symptom in symptoms else 0 for symptom in feature_names])

        prediction = model.predict([input_vector])
        probabilities = model.predict_proba([input_vector])[0]
        
        indices = np.argsort(probabilities)[::-1][:3]
        top_diseases = [(model.classes_[i], probabilities[i] * 100) for i in indices if probabilities[i] > 0]
        
        return top_diseases
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_disease_info(disease):
    """Get information about a disease."""
    disease_info = {
        "Fungal infection": "A fungal infection is a condition caused by fungal growth on or in the body. Treat with antifungal medications. Keep the affected area clean and dry.",
        "Allergy": "An allergy is an immune system response to a foreign substance. Avoid allergens and take antihistamines as needed.",
        "Diabetes": "Diabetes is a metabolic disorder characterized by high blood sugar. Manage with proper diet, exercise, and medication as prescribed.",
        "Common Cold": "A viral infection of the upper respiratory tract. Rest, stay hydrated, and take over-the-counter cold medications for symptom relief.",
        "Pneumonia": "An infection that inflames the air sacs in one or both lungs. Requires antibiotics for bacterial pneumonia, rest, and fluids.",
        "Hypertension": "High blood pressure that can lead to serious health issues. Requires lifestyle changes and possibly medication."
    }
 
    default_info = "Please consult a healthcare professional for accurate diagnosis and treatment information."
    
    return disease_info.get(disease, default_info)

def health_chatbot(query):
    """Enhanced rule-based health chatbot."""
    health_info = {
        "cold": "Common cold symptoms include runny nose, sore throat, cough, and mild fever. Rest, fluids, and over-the-counter medications can help. See a doctor if symptoms persist beyond 10 days.",
        "flu": "Flu symptoms include high fever, body aches, fatigue, and cough. Rest, fluids, and antiviral medications (if prescribed early) can help. Seek medical attention for severe symptoms.",
        "diabetes": "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, and blurred vision. If you suspect diabetes, consult a healthcare provider for proper diagnosis and management.",
        "hypertension": "Hypertension (high blood pressure) often has no symptoms but can cause headaches, shortness of breath, and nosebleeds. Regular check-ups are important for early detection.",
        "covid": "COVID-19 symptoms may include fever, cough, fatigue, loss of taste or smell, and shortness of breath. If you suspect COVID-19, get tested and follow isolation guidelines.",
        "nutrition": "Proper nutrition includes a balanced diet with fruits, vegetables, lean proteins, and whole grains. Limit processed foods, sugars, and excessive salt.",
        "exercise": "Regular exercise, at least 150 minutes per week of moderate activity, is recommended for maintaining good health. Consult a doctor before starting a new exercise routine.",
        "headache": "Headaches can be caused by stress, dehydration, lack of sleep, or underlying conditions. Rest, hydration, and over-the-counter pain relievers can help mild headaches.",
        "stress": "Stress management techniques include deep breathing, meditation, physical activity, and maintaining social connections. Seek professional help for persistent stress or anxiety.",
        "sleep": "Good sleep hygiene includes consistent sleep schedule, comfortable environment, limiting screen time before bed, and avoiding caffeine and alcohol close to bedtime.",
        "fever": "Fever is often a sign that your body is fighting an infection. Rest, stay hydrated, and use over-the-counter fever reducers if needed. Seek medical attention for high fevers (above 103°F/39.4°C).",
        "allergy": "Allergies are immune system reactions to substances that are typically harmless. Symptoms include sneezing, itching, and rashes. Avoid triggers and consider antihistamines for relief."
    }

    query_lower = query.lower()

    for key, info in health_info.items():
        if key in query_lower:
            return info

    if hasattr(st.session_state, 'all_symptoms'):
        for symptom in st.session_state.all_symptoms:
            symptom_lower = symptom.lower().replace('_', ' ')
            if symptom_lower in query_lower:
                return f"'{symptom}' could be a symptom of several conditions. You can use the Symptom Checker tab to check possible conditions based on your symptoms."

    return "I don't have specific information on that health topic. For medical advice, please consult a healthcare professional."

def medicine_reminder_ui():
    """User interface for medicine reminders."""
    st.header("⏰ Medicine Reminder")
    
    with st.form("reminder_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            medicine = st.text_input("Medicine Name")
            dosage = st.text_input("Dosage (e.g., 500mg)")
        
        with col2:
            phone = st.text_input("Phone Number (with country code)")
            reminder_time = st.time_input("Reminder Time")
        
        submit = st.form_submit_button("Set Reminder")
        
        if submit:
            if medicine and phone:
                message_body = f"REMINDER: Time to take {medicine} {dosage} at {reminder_time.strftime('%I:%M %p')}"
                
                st.success(f"Reminder set for {medicine} at {reminder_time.strftime('%I:%M %p')}")
                st.info("In the full app, this would send an SMS reminder to your phone.")

                st.session_state.reminders.append({
                    'medicine': medicine,
                    'dosage': dosage,
                    'time': reminder_time.strftime('%I:%M %p'),
                    'phone': phone
                })
            else:
                st.error("Please enter both medicine name and phone number")
    
    if st.session_state.reminders:
        st.subheader("Your Reminders")
        for i, reminder in enumerate(st.session_state.reminders):
            st.write(f"{i+1}. {reminder['medicine']} {reminder['dosage']} at {reminder['time']}")
        
        if st.button("Clear All Reminders"):
            st.session_state.reminders = []
            st.success("All reminders cleared!")

with tab1:
    st.header("🔧 Setup and Model Training")
    
    st.subheader("1. Load Dataset")
    if st.button("Load Data") or st.session_state.data_loaded:
        symptoms_data = load_data()
        
    st.subheader("2. Train or Load Model")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train New Model"):
            if 'data_loaded' in st.session_state and st.session_state.data_loaded:
                if 'symptoms_data' in locals() and symptoms_data is not None:
                    model, feature_names = train_model(symptoms_data)
                else:
                    st.error("Please load data first")
            else:
                st.error("Please load data first")
    
    with col2:
        if st.button("Load Existing Model"):
            model, feature_names = load_trained_model()


with tab2:
    st.header("🩺 Symptom Checker")
    
    if st.session_state.model_trained:
        if 'all_symptoms' in st.session_state:
            selected_symptoms = st.multiselect(
                "Select your symptoms:", 
                st.session_state.all_symptoms
            )
        else:
            selected_symptoms = st.multiselect(
                "Select your symptoms:", 
                st.session_state.feature_names
            )
        
        if st.button("Check Diagnosis"):
            if selected_symptoms:
                predictions = predict_disease(
                    selected_symptoms, 
                    st.session_state.model, 
                    st.session_state.feature_names
                )
                
                if predictions:
                    st.subheader("Possible Conditions:")
                    for disease, confidence in predictions:
                        st.success(f"{disease} - Confidence: {confidence:.2f}%")
                        
                        with st.expander(f"About {disease}"):
                            st.write(get_disease_info(disease))
                    
                    st.warning("This is an AI prediction and should not replace professional medical advice.")
                else:
                    st.error("No matching conditions found. Please select different symptoms or consult a healthcare professional.")
            else:
                st.error("Please select at least one symptom")
    else:
        st.info("Please load or train a model in the Setup tab first.")

with tab3:
    st.header("💬 Health Chatbot")
    
    query = st.text_input("Ask a health question:")
    
    if st.button("Get Health Information") or query:
        if query:
            response = health_chatbot(query)
            
            st.markdown("### Response:")
            st.markdown(f"> {response}")
            
            st.info("For specific medical advice, please consult a healthcare professional.")
        else:
            st.error("Please enter a health question")

with tab4:
    medicine_reminder_ui()

st.markdown("---")
st.markdown("**MedBuddy AI** - Created for educational purposes. Not a substitute for professional medical advice.")