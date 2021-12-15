import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)

joblib.dump(reg, "linreg.joblib")

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([0, 0, 1, 1])
clf = LogisticRegression().fit(X, y)

joblib.dump(clf, "logreg.joblib")
