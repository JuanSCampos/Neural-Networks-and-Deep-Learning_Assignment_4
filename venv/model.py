# Import necessary libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the breast cancer dataset from sklearn
data = load_breast_cancer()
# Convert the dataset to a DataFrame for easier handling
df = pd.DataFrame(data=data.data, columns=data.feature_names)
# Add the target column to the DataFrame
df['target'] = data.target

# Check for missing values and handle them if necessary
df.isnull().sum()  # Check for any missing values

# Split the dataset into features and target
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target variable

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

from sklearn.feature_selection import SelectKBest, f_classif

# Select the top 10 features based on their statistical significance
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)  # Fit and transform the training set
X_test_selected = selector.transform(X_test)  # Transform the test set using the fitted selector

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
model = MLPClassifier(max_iter=1000)  # Artificial Neural Network with a maximum of 1000 iterations

# Define the parameter grid for Grid Search
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],  # Different architectures for hidden layers
    'activation': ['tanh', 'relu'],  # Activation functions to be tested
    'solver': ['sgd', 'adam'],  # Solvers for weight optimization
    'alpha': [0.0001, 0.05],  # Regularization parameter
    'learning_rate': ['constant','adaptive'],  # Learning rate schedule
}

# Set up Grid Search with 3-fold cross-validation
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
grid_search.fit(X_train_selected, y_train)  # Fit the grid search to the training data

# Display the best parameters found by Grid Search
print("Best parameters found: ", grid_search.best_params_)

# Train the best model found by Grid Search
best_model = grid_search.best_estimator_  # Get the best model
best_model.fit(X_train_selected, y_train)  # Train the model using the selected features

# Evaluate the model on the test set
from sklearn.metrics import classification_report, accuracy_score

y_pred = best_model.predict(X_test_selected)  # Make predictions on the test set
# Print the accuracy of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Save the feature selector and the trained model to files for later use
import pickle
with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
with open('breast_cancer_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)