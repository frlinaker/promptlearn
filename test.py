from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from promptlearn import PromptClassifier

# 1. Load the dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target # Target labels

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and train the K-Nearest Neighbors classifier
pcl = PromptClassifier()
pcl.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = pcl.predict(X_test)

# 5. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Make a prediction on a new, unseen data point
new_data_point = [[5.1, 3.5, 1.4, 0.2]] # Example features for a new data point
prediction = pcl.predict(new_data_point)
print(f"Prediction for new data point: {iris.target_names[prediction[0]]}")