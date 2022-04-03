import random

import gensim
import pandas as pd
from xml.dom import minidom
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer

word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
seType_dict = {'STATE': 0, 'EVENT': 1, 'REPORT': 2, 'GENERIC_SENTENCE': 3, 'GENERALIZING_SENTENCE': 4, 'QUESTION': 5, 'IMPERATIVE': 6}


def helper(filename_list, stanford_nlp, if_do_embedding, tokenizer):
    print("len(filename_list): ", len(filename_list))
    data_list = []
    i = 0
    for filename in filename_list[:]:
        i += 1
        if i % 10 == 0:
            print("already deal {0} file".format(i))
            # break
        DOMTree = minidom.parse('data/MASC_Wikipedia/annotations_xml/' + filename[:len(filename) - 3] + "xml")
        document = DOMTree.documentElement
        segments = document.getElementsByTagName('segment')

        for segment in segments:
            text = segment.getElementsByTagName('text')[0].childNodes[0].data
            annotation = segment.getElementsByTagName('annotation')[0]
            if annotation.hasAttribute('seType'):  # gold has seType
                seType = annotation.getAttribute('seType')
                if seType in seType_dict.keys():
                    if if_do_embedding:  # for BiLSTM
                        data_list.append([text, seType_dict[seType], from_sentence_2_word_embeddings_list(text, stanford_nlp, word2vec_vocab)])
                    else:  # for BERT
                        data_list.append([text, seType_dict[seType], tokenizer(text, return_tensors="pt").input_ids.cuda()])
    return data_list


def get_data(if_do_embedding, stanford_path, random_seed):
    print("start to get stanford")
    if if_do_embedding:
        stanford_nlp = StanfordCoreNLP(stanford_path)  # default english, useless for BERT
    else:  # BERT
        stanford_nlp = None
    print("already get stanford")

    print('start get bert_tokenizer')
    tokenizer = BertTokenizer.from_pretrained('pre_train')
    print('get bert tokenizer')

    print("if_do_embedding: ", if_do_embedding)

    print("start read data catalogue")
    train_test_split_csv = pd.read_csv('data/MASC_Wikipedia/train_test_split.csv', '\t')[:]

    test_filename_list = []  # get [filename1.txt, ......]
    train_filename_list = []
    for i in range(len(train_test_split_csv)):
        if train_test_split_csv.iloc[i]['fold'] == "train":
            train_filename_list.append(train_test_split_csv.iloc[i]['category_filename'])
        if train_test_split_csv.iloc[i]['fold'] == "test":
            test_filename_list.append(train_test_split_csv.iloc[i]['category_filename'])

    print("start get train data")
    train_data_list = helper(train_filename_list, stanford_nlp, if_do_embedding, tokenizer)
    print("len(train_valid_data_list): ", len(train_data_list))
    test_data_list = helper(test_filename_list, stanford_nlp, if_do_embedding, tokenizer)
    print("len(test_data_list): ", len(test_data_list))

    random.seed(random_seed)
    random.shuffle(train_data_list)

    # if stanford_nlp:  # open a stanford_nlp just now
    #     stanford_nlp.close()

    valid_list_len = len(test_data_list)  # choose the same len as test
    train_list_len = len(train_data_list) - valid_list_len  # from train get valid

    return train_data_list[:train_list_len], train_data_list[train_list_len:], test_data_list

