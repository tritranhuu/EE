import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def train_and_test_with_sgd(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train+y_test)
    classes = classes.tolist()
    print(classes)

    sgd = SGDClassifier()
    sgd.partial_fit(X_train, y_train, classes=classes)

    new_class = list(set(classes)- set(['O']))
    print(classification_report(y_pred=sgd.predict(X_test), y_true=y_test, labels=new_class))