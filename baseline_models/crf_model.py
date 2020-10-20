import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from vncorenlp import VnCoreNLP

class CRFModel():
    def __init__(self, sent):
        self.model = CRF(algorithm='lbfgs',
                        c1=0.1,
                        c2=0.1,
                        max_iterations=100,
                        all_possible_transitions=False)
        self.X = np.array([self.sent2features(s) for s in sent])
        self.y = np.array([self.sent2labels(s) for s in sent])

    def report(self):
        pred = cross_val_predict(estimator=self.model, X=self.X[:3000], y=self.y[:3000], cv=5)
        report = flat_classification_report(y_pred=pred, y_true=self.y[:3000])
        print(report)
    
    def train(self):
        self.model.fit(self.X[:3200],self.y[:3200])
    
    def score(self):
        pred = self.model.predict(self.X[3200:])
        return flat_classification_report(y_pred=pred, y_true=self.y[3200:])
    
    def pred(self, sent):
        # annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
        # sent_tokenized = annotator.tokenize(sent)[0]
        sent_tokenized = sent.split(" ")
        sentence = {
            "words": sent_tokenized,
            "anno": []
        }
        features = [self.sent2features(sentence)]
        return self.model.predict(features)


    def word2feature(self, sent, i):
        word = sent['words'][i]
        # tag = sent['anno'][i]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            # 'tag': tag,
            # 'tag[:2]': tag[:2]
        }
        if i >0:
            word1 = sent['words'][i-1]
            # tag1 = sent['anno'][i-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isdigit()': word1.isdigit(),
                # '-1:tag': tag1,
                # '-1:tag[:2]': tag[:2]
            })
        else:
            features['BOS'] = True
        if i<len(sent['words'])-1:
            word1 = sent['words'][i+1]
            # tag1 = sent['anno'][i+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isdigit()': word1.isdigit(),
                # '+1:tag': tag1,
                # '+1:tag[:2]': tag[:2]
            })
        else:
            features['EOS'] = True
        return features

    def sent2features(self, sent):
        return np.array([self.word2feature(sent, i) for i in range(len(sent['words']))])
        
    def sent2labels(self, sent):
        return np.array(sent['labels'])
        
    def sent2tokens(self, sent):
        return sent['words']
        
