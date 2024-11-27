# Developing the Streamlit App

# Importing the necessary libraries
import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset to get feature names
data = load_breast_cancer()

# Load the feature selector and trained model from files
with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('breast_cancer_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Streamlit App Design
st.set_page_config(page_title="Breast Cancer Prediction App", layout="centered", initial_sidebar_state="expanded")

# App title and description
st.title('🌸 Breast Cancer Prediction App')
st.markdown("This application uses machine learning to predict whether a tumor is benign or malignant based on user inputs. 💻📊")

# Sidebar header for input features
st.sidebar.header('Input Features')
st.sidebar.markdown('Adjust the features below to see the prediction results.')

# User input features - create sliders for each selected feature
user_input = []
feature_indices = selector.get_support(indices=True)  # Get indices of selected features
for feature_index in feature_indices:
    feature_name = data.feature_names[feature_index]  # Get feature name
    # Create a slider for each feature in the sidebar
    user_value = st.sidebar.slider(
        f"{feature_name}",
        float(data.data[:, feature_index].min()),
        float(data.data[:, feature_index].max()),
        float(data.data[:, feature_index].mean())
    )
    user_input.append(user_value)  # Append user input value to the list

# Convert user input to a NumPy array
user_input = np.array(user_input).reshape(1, -1)

# Predict using the trained model
if st.sidebar.button('Predict 🧠'):
    prediction = best_model.predict(user_input)  # Make a prediction based on user input
    prediction_proba = best_model.predict_proba(user_input)  # Get prediction probabilities

    # Display the prediction result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('### 🔴 The tumor is **Malignant**')
    else:
        st.write('### 🟢 The tumor is **Benign**')

    # Display prediction probability for each class
    st.write("#### Prediction Probability:")
    st.write(f"- Malignant: {prediction_proba[0][1] * 100:.2f}%")
    st.write(f"- Benign: {prediction_proba[0][0] * 100:.2f}%")

# Styling and information about the app
st.markdown("---")
st.markdown("### About This App")
st.markdown(
    "This app was created to help demonstrate how machine learning can be used in medical applications to assist in diagnosis. The model is trained on the Breast Cancer dataset from sklearn.")
st.markdown("Feel free to play around with the input features to see how the predictions change!")

# Footer with author information
st.markdown("---")
st.write("Made by Juan Campos")
st.write("GitHub Repository: https://github.com/JuanSCampos/Neural-Networks-and-Deep-Learning_Assignment_4.git")