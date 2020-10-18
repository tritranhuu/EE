import numpy as np
import pickle as pkl
import sys
import itertools

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def find_max_length(sents):
    max_length = 0
    temp_length = 0

    for sent in sents:
        if len(sent['words']) > max_length:
            max_length = len(sent['words'])
    
    return max_length


def entity(tag):
    one_hot = np.zeros(14)
    if tag == 'PER' :
        one_hot[0] = 1
    elif tag == 'GPE' :
        one_hot[1] = 1
    elif tag == 'CRIME' :
        one_hot[2] = 1
    elif tag == 'TIME' :
        one_hot[3] = 1
    elif tag == 'ORG' :
        one_hot[4] = 1
    elif tag == 'JOB' :
        one_hot[5] = 1
    elif tag == 'LOC' :
        one_hot[6] = 1
    elif tag == 'FAC' :
        one_hot[7] = 1
    elif tag == 'MONEY' :
        one_hot[8] = 1
    elif tag == 'SEN' :
        one_hot[9] = 1
    elif tag == 'WEA' :
        one_hot[10] = 1
    elif tag == 'VEH' :
        one_hot[11] = 1
    elif tag == 'NUM' :
        one_hot[12] = 1
    else:
        one_hot[13] = 1
    return one_hot

def get_event_tags():
    events = {
        "Life" : ['Die', 'Marry', 'Divorce', 'Injure', 'Be-born'],
        "Movement" : ['Transport'],
        "Transaction": ['Transfer-ownership', 'Transfer-money'],
        "Business" : ['Start-org', 'Merge-org', 'Declare-bankruptcy', 'End-org'],
        "Conflict" : ['Attack', 'Demonstrate'],
        "Contact" : ['Phone-write', 'Meet'],
        "Personell" : ['Elect', 'End-position', 'Nominate', 'Start-position'],
        "Justice" : ['Pardon', 'Convict', 'Sentence', 'Appeal', 'Sue', 'Arrest-jail', 'Release-parole', 'Trial-hearing', 'Charge-indict', 'Fine', 'Execute', 'Extradite', 'Arquit']
    }
    sub_events = [v for k, v in events.items()]
    sub_events = list(itertools.chain(*sub_events))
    sub_events = np.append(sub_events, ['O'])
    
    return sub_events

def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])


def get_input(model, word_dim, input_data, output_embed, output_tag, sentence_length = 30):  
    
    event_labels = get_event_tags()
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    int_encoded = label_encoder.fit_transform(event_labels)
    print(int_encoded)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoder.fit(int_encoded)
    
    word = []
    tag = []
    sentence = []
    sentence_tag = []

    if sentence_length == -1:
        max_sentence_length = find_max_length(input_data)
    else:
        max_sentence_length = sentence_length
    sentence_length = 0
    for sent in input_data:
        sentence_length = len(sent['words'])
        for i in range(max_sentence_length):
            if i < sentence_length:
                temp = model[sent['words'][i]]
                temp = np.append(temp, entity(sent['entities'][i]))
                # temp = np.append(temp, capital(sent['words'][i]))
                word.append(temp)
                tag_one_hot = onehot_encoder.transform([label_encoder.transform([sent['labels'][i]])]).toarray()[0]
                tag.append(tag_one_hot)
            else:
                temp = np.array([0 for _ in range(word_dim + 14)])
                word.append(temp)
                x = np.zeros(34)
                tag.append(x)
        sentence.append(word)
        sentence_tag.append(tag)
        word = []
        tag = []
    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))




# if __name__ == "__main__":

    