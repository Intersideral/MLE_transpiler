import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)

joblib.dump(reg, "linreg.joblib")

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, max_iter=150).fit(X, y)

joblib.dump(clf, "logreg.joblib")
