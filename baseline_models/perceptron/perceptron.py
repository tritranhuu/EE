import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

def train_and_test_with_perceptron(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train+y_test)
    classes = classes.tolist()
    print(classes)

    per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    per.partial_fit(X_train, y_train, classes=classes)

    print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=classes))