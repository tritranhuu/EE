from prepare_data.data_utils import DataProcessor
from baseline_models.crf.crf_model import CRFModel

if __name__ == "__main__":
    data = DataProcessor('data/dev/')
    data.format_to_file('data/dev.txt')
    # crf = CRFModel(data.sentences)
    # print(crf.X.shape, crf.y.shape)
    # # print(crf.cross_val_predict())

    # crf.train()
    # print(crf.score())
    # print(crf.pred("he Daily Planet Ltd , about to become the first brothel to list on the Australian Stock Exchange , plans to follow up its May Day launching by opening a `` sex Disneyland '' here , the Melbourne-based bordello announced Wednesday"))

