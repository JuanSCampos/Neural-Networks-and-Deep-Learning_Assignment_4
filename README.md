# 🌸 Breast Cancer Prediction Project

## 📝 Overview

This project is an interactive web application designed to predict whether a breast tumor is benign or malignant. Using a machine learning model trained on the Breast Cancer dataset from sklearn, the app allows users to input various medical parameters to receive a prediction. This project demonstrates how machine learning can be utilized in medical applications to assist in diagnosis.

## 🚀 Features

- Predict whether a breast tumor is Benign or Malignant.

- Interactive user interface to input medical features.

- Real-time prediction and probability visualization.

- Easy-to-use and visually appealing design.

## ⚙️ Installation & Setup

To get started, follow these instructions to set up the project on your local machine.

**1. Clone the Repository**

*git clone <repository_url>*
*cd breast_cancer_analysis*

**2. Set Up a Virtual Environment**

Create and activate a virtual environment for the project.

*python -m venv venv*
*source venv/bin/activate  # On Windows use `venv\Scripts\activate`*

**3. Install Requirements**

Install the required Python packages from requirements.txt.

*pip install -r requirements.txt*

**4. Run the Application**

Use Streamlit to run the app locally.

*streamlit run app.py*

## 📂 Project Structure

- model.py: Contains the code for data preprocessing, feature selection, and model training.

- app.py: Streamlit application for user interaction and predictions.

- selector.pkl and breast_cancer_model.pkl: Saved feature selector and trained model files.

## 🧠 Model Information

The model used in this project is an Artificial Neural Network (*MLPClassifier*), optimized using Grid Search CV. The best parameters found include different hidden layer architectures, activation functions, and solvers.

## 🛠️ Technologies Used

- Python

- Streamlit for building the web app

- scikit-learn for machine learning and data processing

- Pandas for data manipulation

## 🎮 Usage

1. Launch the app using Streamlit.

2. Use the sliders in the sidebar to input the medical features.

3. Click the **Predict** button to view whether the tumor is benign or malignant, along with the prediction probability.

## 🎨 App Design

- The app includes visually appealing elements like emojis and an intuitive sidebar for user input.

- Real-time feedback is provided to users in the form of predictions and probabilities.

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 👤 Authors

Made by **Juan Sebastian Campos Perez**

## 🙏 Acknowledgements

- scikit-learn for providing the Breast Cancer dataset.

- Streamlit for the amazing framework to create web apps with Python.

## 📧 Contact

If you have any questions or suggestions, feel free to reach out!

