import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

def train_and_test_with_perceptron(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train)
    classes = classes.tolist()
    print(classes)

    per = Perceptron(verbose=10, n_jobs=-1)
    per.partial_fit(X_train, y_train, classes=classes)

    new_class = list(set(classes)- set(['O']))
    print(new_class)
    print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_class))