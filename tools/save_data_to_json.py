import json
import gensim
import pandas as pd
from xml.dom import minidom
from transformers import BertTokenizer
from stanfordcorenlp import StanfordCoreNLP
from tools.from_sentence_2_word_embeddings_list import from_sentence_2_word_embeddings_list


word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('resource/GoogleNews-vectors-negative300.bin', binary=True)
seType_dict = {'STATE': 0, 'EVENT': 1, 'REPORT': 2, 'GENERIC_SENTENCE': 3, 'GENERALIZING_SENTENCE': 4, 'QUESTION': 5, 'IMPERATIVE': 6, 'no': 7}


def helper(filename_list, stanford_nlp, if_do_embedding, tokenizer, save_json):
    # all_data_list = []  # [[paragraph0, label_list0, helpful_label_list_len], [paragraph1, label_list1], ]

    label_list = []  # ['STATE', 'EVENT', ]
    segment_list = []  # [sentence0, sentence1, ]
    helpful_label_list_len = 0  #
    segment_embeddings_list = []

    i = 0
    for filename in filename_list[:]:
        if i % 10 == 0:
            print("already deal {0} file".format(i))
        i += 1

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
                    this_paragraph_data = [segment_list, [seType_dict[label] for label in label_list], helpful_label_list_len, segment_embeddings_list]
                    # this_paragraph_data = json.dumps(this_paragraph_data)
                    json.dump(this_paragraph_data, save_json)
                    save_json.write('\n')
                    save_json.flush()
                label_list = []
                segment_list = []
                helpful_label_list_len = 0
                segment_embeddings_list = []

            text = segment.getElementsByTagName('text')[0].childNodes[0].data
            annotation = segment.getElementsByTagName('annotation')[0]  # the first annotation is gold
            if annotation.hasAttribute('seType'):  # gold has seType
                seType = annotation.getAttribute('seType')
                if seType not in seType_dict.keys():
                    seType = 'no'
                else:
                    helpful_label_list_len += 1
            else:
                seType = 'no'

            segment_list.append(text)
            label_list.append(seType)

            if if_do_embedding:  # for BiLSTM
                segment_embeddings_list.append(from_sentence_2_word_embeddings_list(text, stanford_nlp, word2vec_vocab))
            else:  # for BERT
                segment_embeddings_list.append(tokenizer(text, return_tensors="pt").input_ids.cuda())


def get_and_save(if_do_embedding, stanford_path):
    print("if_do_embedding: ", if_do_embedding)

    if if_do_embedding:
        print("start to get stanford")
        tokenizer = None
        stanford_nlp = StanfordCoreNLP(stanford_path)  # default english, useless for BERT
        print("already get stanford")
    else:  # BERT
        stanford_nlp = None
        print('start get bert_tokenizer')
        tokenizer = BertTokenizer.from_pretrained('pre_train')
        print('get bert tokenizer')

    print("start read data catalogue")
    train_test_split_csv = pd.read_csv('data/MASC_Wikipedia/train_test_split.csv', '\t')[:]

    test_filename_list = []  # get [filename1.json, ......]
    train_filename_list = []
    for i in range(len(train_test_split_csv)):
        if train_test_split_csv.iloc[i]['fold'] == "train":
            train_filename_list.append(train_test_split_csv.iloc[i]['category_filename'])
        if train_test_split_csv.iloc[i]['fold'] == "test":
            test_filename_list.append(train_test_split_csv.iloc[i]['category_filename'])

    print("\nstart get train data")
    train_data_json = open('data/train_data_json_300.json', 'w')
    helper(train_filename_list, stanford_nlp, if_do_embedding, tokenizer, train_data_json)
    train_data_json.close()

    print("\nstart get test data")
    test_data_json = open('data/test_data_json_300.json', 'w')
    helper(test_filename_list, stanford_nlp, if_do_embedding, tokenizer, test_data_json)
    test_data_json.close()

    if if_do_embedding:  # open a stanford_nlp just now
        stanford_nlp.close()