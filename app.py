import streamlit as st
import joblib

# Load your trained model
model = joblib.load('./random_forest_model.joblib')

# Mappings for categorical variables
mappings = {
    'sex': {'F': 0, 'M': 1},
    'address': {'U': 0, 'R': 1},
    'family_size': {'LE3': 0, 'GT3': 1},
    'parents_status': {'T': 0, 'A': 1},
    'mother_job': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
    'father_job': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
    'reason': {'home': 0, 'reputation': 1, 'course': 2, 'other': 3},
    'guardian': {'mother': 0, 'father': 1, 'other': 2},
    'school_support': {'yes': 0, 'no': 1},
    'family_support': {'yes': 0, 'no': 1},
    'paid_classes': {'yes': 0, 'no': 1},
    'activities': {'yes': 0, 'no': 1},
    'nursery': {'yes': 0, 'no': 1},
    'desire_higher_edu': {'yes': 0, 'no': 1},
    'internet': {'yes': 0, 'no': 1},
    'romantic': {'yes': 0, 'no': 1},
}

# Function for encoding categorical variables


def encode_categorical(key, value):
    # Returns -1 if the value is not found in the mapping
    return mappings[key].get(value, -1)

# Function to preprocess and convert input data to the required format


def preprocess_input_data(input_data):
    processed_data = []
    for key, value in input_data.items():
        if key in mappings:
            processed_value = encode_categorical(key, value)
        else:
            processed_value = float(value)
        processed_data.append(processed_value)
    return processed_data

# Streamlit UI


def main():
    st.title("Student Performance Prediction")

    # Define the input fields
    user_input = {}
    categorical_keys = list(mappings.keys())
    numerical_keys_info = {
        'age': (15, 22),
        'mother_education': (0, 4),
        'father_education': (0, 4),
        'study_time': (1, 4),
        'failures': (0, 4),
        'family_quality': (1, 5),
        'free_time': (1, 5),
        'go_out': (1, 5),
        'weekday_alcohol_usage': (1, 5),
        'weekend_alcohol_usage': (1, 5),
        'health': (1, 5),
        'absences': (0, 93),
        'period1_score': (0, 20),
        'period2_score': (0, 20),
        'final_score': (0, 20)
    }

    for key in categorical_keys:
        options = list(mappings[key].keys())
        user_input[key] = st.selectbox(f"Select {key}", options=options)

    for key, (min_val, max_val) in numerical_keys_info.items():
        user_input[key] = st.slider(f"Select {key}", min_val, max_val, min_val)

    if st.button("Predict"):
        processed_input = preprocess_input_data(user_input)
        input_sample = [processed_input]
        prediction = model.predict(input_sample)
        st.success(f"The predicted result is: {prediction[0]}")


if __name__ == "__main__":
    main()
