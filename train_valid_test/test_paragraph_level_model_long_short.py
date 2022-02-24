import torch
import numpy as np
from tools.print_evaluation_result import print_evaluation_result


def helper(test_data_list, model, with_raw_text=False):
    model.eval()
    useful_target_Y_list = []
    useful_predict_Y_list = []
    with torch.no_grad():
        for test_data in test_data_list:
            sentences_list = []
            gold_labels_list = []
            raw_sentences_list = []
            for sentence, label, raw_sentence in zip(test_data[3], test_data[1], test_data[0]):
                # for BiLSTM, sentence is words_embeddings_list
                # for BERT, sentence is words_ids_list
                if label != 7:
                    gold_labels_list.append(label)
                    sentences_list.append(sentence)
                    raw_sentences_list.append(raw_sentence)

            if with_raw_text:
                pre_labels_list, _ = model.forward([sentences_list, raw_sentences_list], gold_labels_list)  # sentence_num * 7
            else:
                pre_labels_list, _ = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

            for i in range(len(pre_labels_list)):
                useful_target_Y_list.append(gold_labels_list[i])
                useful_predict_Y_list.append(pre_labels_list[i])

    print('test print ')

    f1_score, acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)

    return f1_score, acc


def long_short_get(data_list, model=None, valid=False, with_raw_text=False):

    length_list = []
    length_list_with_overlap = []
    data_sort_list = dict()
    for data in data_list:
        if data[2] not in length_list:
            data_sort_list[data[2]] = [data]
            length_list.append(data[2])  # data[2] is paragraph_len
        else:
            data_sort_list[data[2]].append(data)

        length_list_with_overlap.append(data[2])

    print("np.mean(np.array(length_list_with_overlap)): ", np.mean(np.array(length_list_with_overlap)))
    print("np.median(np.array(length_list_with_overlap)): ", np.median(np.array(length_list_with_overlap)))

    length_list = sorted(length_list)
    for length in length_list:
        print(length, len(data_sort_list[length]))

    if valid == False:  # for test
        for l in range(1, 15):
            if l in length_list:
                print("\n\n-------------------- test data.sen_num == {} ---------------".format(l))
                test_data_list = data_sort_list[l]
                helper(test_data_list, model, with_raw_text)

        print("\n\n-------------------- test data.sen_num == lager than 15 ---------------")
        test_data_list = []
        for l in range(15, 211):
            if l in length_list:
                test_data_list += data_sort_list[l]
        helper(test_data_list, model, with_raw_text)

