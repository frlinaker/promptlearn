#!/usr/bin/env python

"""
Example: Using PromptClassifier on the Iris dataset with pandas and scikit-learn.
"""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)

# 1. Load the dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore
y = pd.Series(iris.target, name="species")  # type: ignore

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and train the PromptClassifier
pcl = PromptClassifier(verbose=True)  # enable verbose logging
pcl.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = pcl.predict(X_test)

# 5. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Make a prediction on a new, unseen data point
new_data_point = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)  # type: ignore
prediction = pcl.predict(new_data_point)
print(f"Prediction for new data point: {iris.target_names[prediction[0]]}")  # type: ignore
print("(The best prediction would be 'setosa' for this input.)")
