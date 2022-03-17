import os
import torch
import random


def load_data():
    print('Loading Data...')
    with open(os.path.join(os.getcwd(), 'data/masc_paragraph_addposnerembedding_dictformat_with_raw_sentence.pt'), 'rb') as outfile:
        data = torch.load(outfile)
        outfile.close()

    train_doc_list = []
    test_doc_list = []
    with open('data/MASC_Wikipedia/train_test_split.csv', 'r') as f:
        for line in f.readlines():
            filename, genre, index, train_test = line.split()
            filename = filename[:-4]
            if train_test == 'train':
                train_doc_list.append(filename)
            elif train_test == 'test':
                test_doc_list.append(filename)

    train_X_eos_list = []
    train_X_label_length_list = []
    train_X = []
    train_X_raw_sentence = []
    train_Y = []
    for filename in train_doc_list:
        doc_x, doc_y = data[filename]
        train_X += doc_x[0]
        train_X_label_length_list += doc_x[1]
        train_X_eos_list += doc_x[2]
        train_X_raw_sentence += doc_x[4]
        train_Y += doc_y

    test_X_eos_list = []
    test_X_label_length_list = []
    test_X = []
    test_Y = []
    test_X_raw_sentence = []
    for filename in test_doc_list:
        doc_x, doc_y = data[filename]
        test_X += doc_x[0]
        test_X_label_length_list += doc_x[1]
        test_X_eos_list += doc_x[2]
        test_X_raw_sentence += doc_x[3]
        test_Y += doc_y

    return train_X, train_X_label_length_list, train_X_eos_list, train_Y, test_X, test_X_label_length_list, test_X_eos_list, test_Y, train_X_raw_sentence, test_X_raw_sentence


def helper(train_X, train_Y, train_X_eos_list, train_X_raw_sentence):

    data_list = []
    for paragraph_X, paragraph_Y, paragraph_X_eos_list, raw_sentence_list in zip(train_X, train_Y, train_X_eos_list, train_X_raw_sentence):
        sentence_X_list = []
        gold_labels_list = []
        useful_gold_labels_num = 0
        for Y in paragraph_Y:
            if 1 in Y:
                gold_labels_list.append(torch.argmax(Y))
                useful_gold_labels_num += 1
            else:
                gold_labels_list.append(7)
        start = 0
        for eos in paragraph_X_eos_list:
            sentence_X_list.append(paragraph_X[0, start:eos, :])
            start = eos
        data_list.append([raw_sentence_list, gold_labels_list, useful_gold_labels_num, sentence_X_list])

    return data_list


def re_load(random_seed):
    print('use 343 from author')
    train_X, train_X_label_length_list, train_X_eos_list, train_Y, test_X, test_X_label_length_list, test_X_eos_list, test_Y, train_X_raw_sentence, test_X_raw_sentence = load_data()

    print("len(test_X): ", len(test_X))
    print("len(train_X): ", len(train_X))

    test_data_list = helper(test_X, test_Y, test_X_eos_list, test_X_raw_sentence)
    train_data_list = helper(train_X, train_Y, train_X_eos_list, train_X_raw_sentence)

    random.seed(random_seed)
    random.shuffle(train_data_list)

    return train_data_list, test_data_list