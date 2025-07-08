from promptlearn import PromptClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)

clf = PromptClassifier(chunk_threshold=200, max_chunks=5)
print(clf.get_params())
# Should print all constructor params and their values

# 1. Load the dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore
y = pd.Series(iris.target, name="species")  # type: ignore

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {"chunk_threshold": [100, 200]}
search = GridSearchCV(clf, param_grid, cv=2)
search.fit(X, y)
print(search.best_params_)
