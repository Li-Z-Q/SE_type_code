import json
import random


def paragraph_helper(json_file):
    data_list = []
    for line in json_file.readlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list


def get_paragraph_data(dim, random_seed):
    print('getting paragraph data')
    test_data_json = open('data/test_data_json_' + str(dim) + '.json', 'r')
    train_data_json = open('data/train_data_json_' + str(dim) + '.json', 'r')

    print('will get test')
    test_data_list = paragraph_helper(test_data_json)
    print('will get train')
    train_data_list = paragraph_helper(train_data_json)

    random.seed(random_seed)
    random.shuffle(train_data_list)

    print("len(test_data_list): ", len(test_data_list))
    print("len(train_data_list): ", len(train_data_list))

    return train_data_list, test_data_list


# def sentence_helper(json_file):
#     print('sentence helper')
#     data_list = []
#     i = 0
#     for line in json_file.readlines():
#         data = json.loads(line)
#         for sentence, gold_label, word_embeddings_list in zip(data[0], data[1], data[3]):
#             if gold_label != 7:  # is not 'no'
#                 data_list.append([sentence, gold_label, word_embeddings_list])
#     return data_list
#
#
# def get_sentence_data(random_seed):
#     print('getting sentence data')
#     test_data_json = open('data/test_data_json.json', 'r')
#     train_data_json = open('data/train_data_json.json', 'r')
#
#     print('will get test')
#     test_data_list = sentence_helper(test_data_json)
#     print('will get train')
#     train_data_list = sentence_helper(train_data_json)
#
#     random.seed(random_seed)
#     random.shuffle(train_data_list)
#
#     print("len(test_data_list): ", len(test_data_list))
#     print("len(train_data_list): ", len(train_data_list))
#
#     return train_data_list, test_data_list

def from_paragraph_to_sentence(paragraph_data_list, random_seed):
    print("from paragraph to sentence")
    data_list = []

    for data in paragraph_data_list:
        for raw_sentence, label, word_embeddings_list in zip(data[0], data[1], data[3]):
            data_list.append([raw_sentence, label, word_embeddings_list])

    print("len(data_list): ", len(data_list))

    random.seed(random_seed)
    random.shuffle(data_list)

    return data_list
