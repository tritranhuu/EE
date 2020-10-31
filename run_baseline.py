# from prepare_data.data_utils import DataProcessor
from sklearn.feature_extraction import DictVectorizer
# from baseline_models.crf.crf_model import CRFModel

from baseline_models.sgd import train_and_test_with_sgd


import pandas as pd

if __name__ == "__main__":
    # data = DataProcessor('data/ace-05-brat/test')
    # data.format_to_file('data/ace-05-brat/csv/test.csv')
    # crf = CRFModel(data.sentences)
    # print(crf.X.shape, crf.y.shape)
    # # print(crf.cross_val_predict())

    # crf.train()
    # print(crf.score())
    # print(crf.pred("he Daily Planet Ltd , about to become the first brothel to list on the Australian Stock Exchange , plans to follow up its May Day launching by opening a `` sex Disneyland '' here , the Melbourne-based bordello announced Wednesday"))

    train = pd.read_csv('data/txt/train.txt', sep='\t', header=None)
    test = pd.read_csv('data/txt/test.txt', sep='\t', header=None)
    
    X_train = train.drop(train.columns[len(train.columns)-1], axis=1)
    y_train = train[train.columns[len(train.columns)-1]].values
    X_test = test.drop(test.columns[len(test.columns)-1], axis=1)
    y_test = test[test.columns[len(test.columns)-1]].values

    v = DictVectorizer(sparse=True)
    v.fit(pd.concat([X_train, X_test]).to_dict('records'))

    X_train = v.transform(X_train.to_dict('records'))
    X_test = v.transform(X_test.to_dict('records'))
    train_and_test_with_sgd(X_train, y_train, X_test, y_test)
