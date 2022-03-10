import random

import pandas as pd
import gensim
from xml.dom import minidom
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list
from train_valid_test.test_paragraph_level_model_long_short import long_short_get
from tools.devide_long_paragraph import devide_long_paragraph_to_short

word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin', binary=True)
seType_dict = {'STATE': 0, 'EVENT': 1, 'REPORT': 2, 'GENERIC_SENTENCE': 3, 'GENERALIZING_SENTENCE': 4, 'QUESTION': 5, 'IMPERATIVE': 6, 'no': 7}


def helper(filename_list, stanford_nlp, if_do_embedding, tokenizer):
    all_data_list = []  # [[paragraph0, label_list0, label_list_len], [paragraph1, label_list1], ]

    label_list = []  # ['STATE', 'EVENT', ]
    segment_list = []  # [sentence0, sentence1, ]
    segment_embeddings_list = []
    label_list_len = 0  #

    i = 0
    for filename in filename_list[:]:
        i += 1
        if i % 10 == 0:
            print("already deal {0} file".format(i))
            # break
        raw_text = open('data/MASC_Wikipedia/raw_text/' + filename, 'r', encoding='utf-8').read()
        DOMTree = minidom.parse('data/MASC_Wikipedia/annotations_xml/' + filename[:len(filename) - 3] + "xml")
        document = DOMTree.documentElement
        segments = document.getElementsByTagName('segment')

        for segment in segments:

            begin = int(segment.getAttribute('begin'))
            pre_begin = begin - 1
            pre_pre_begin = begin - 2

            if begin > 1 and int(ord(raw_text[pre_pre_begin])) == 10 and int(ord(raw_text[pre_begin])) == 10:  # NOT belong to a same paragraph

                if set(label_list) != {'no'}:
                    # label_to_num_list = [seType_dict[label] for label in label_list]
                    # all_data_list.append([segment_list, label_to_num_list, label_list_len, segment_embeddings_list])
                    #
                    label_to_num_list = [seType_dict[label] for label in label_list]
                    if label_list_len < 1000:
                        all_data_list.append([segment_list, label_to_num_list, label_list_len, segment_embeddings_list])
                    else:
                        print("this paragraph.len is: ", label_list_len, "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

                        print("this paragraph.len is: ", label_list_len, "$$$$$ do devide $$$$")
                        devide_long_paragraph_to_short([segment_list, label_to_num_list, label_list_len, segment_embeddings_list],
                                                       all_data_list)  # devide to short adn then append

                label_list = []
                segment_list = []
                segment_embeddings_list = []
                label_list_len = 0

            text = segment.getElementsByTagName('text')[0].childNodes[0].data
            annotation = segment.getElementsByTagName('annotation')[0]  # the first annotation is gold
            if annotation.hasAttribute('seType'):  # gold has seType
                seType = annotation.getAttribute('seType')
                if seType not in seType_dict.keys():
                    seType = 'no'
                else:
                    label_list_len += 1
            else:
                seType = 'no'

            segment_list.append(text)
            label_list.append(seType)
            if if_do_embedding:  # for BiLSTM
                segment_embeddings_list.append(from_sentence_2_word_embeddings_list(text, stanford_nlp, word2vec_vocab))
            else:  # for BERT
                segment_embeddings_list.append(tokenizer(text, return_tensors="pt").input_ids.cuda())
    return all_data_list


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
    print("complete get data, len(test_data_list): ", len(test_data_list))

    random.seed(random_seed)
    random.shuffle(train_data_list)

    # if stanford_nlp:  # open a stanford_nlp just now
    #     stanford_nlp.close()

    valid_list_len = len(test_data_list)  # choose the same len as test
    train_list_len = len(train_data_list) - valid_list_len  # from train get valid

    print("train data long short")
    long_short_get(train_data_list[:train_list_len], model=None, valid=True)
    print("\n\nvalid data long short")
    long_short_get(train_data_list[train_list_len:], model=None, valid=True)

    return train_data_list[:train_list_len], train_data_list[train_list_len:], test_data_list