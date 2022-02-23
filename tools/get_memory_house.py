import random
import pandas as pd
from xml.dom import minidom


seType_dict = {'STATE': 0, 'EVENT': 1, 'REPORT': 2, 'GENERIC_SENTENCE': 3, 'GENERALIZING_SENTENCE': 4, 'QUESTION': 5, 'IMPERATIVE': 6, 'no': 7}

def helper(filename_list):
    print("len(filename_list): ", len(filename_list))
    data_list = []

    for filename in filename_list[:]:
        DOMTree = minidom.parse('data/MASC_Wikipedia/annotations_xml/' + filename[:len(filename) - 3] + "xml")
        document = DOMTree.documentElement
        segments = document.getElementsByTagName('segment')

        for segment in segments:
            text = segment.getElementsByTagName('text')[0].childNodes[0].data
            annotation = segment.getElementsByTagName('annotation')[0]
            if annotation.hasAttribute('seType'):  # gold has seType
                seType = annotation.getAttribute('seType')
                if seType in seType_dict.keys():
                    data_list.append([text, seType_dict[seType]])
    return data_list


def get_memory():
    print('\nwill get memory house')
    print("start read data catalogue")
    train_test_split_csv = pd.read_csv('data/MASC_Wikipedia/train_test_split.csv', '\t')[:]

    train_filename_list = []
    for i in range(len(train_test_split_csv)):
        if train_test_split_csv.iloc[i]['fold'] == "train":
            train_filename_list.append(train_test_split_csv.iloc[i]['category_filename'])

    print("start get train data")
    train_data_list = helper(train_filename_list)
    random.shuffle(train_data_list)

    return train_data_list  # is [[text, data], []]
