import torch
from tools.print_evaluation_result import print_evaluation_result


def test_model(test_data_list, model, with_raw_text=False):
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

    print('\n\ntest print ')

    f1_score, acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)

    return f1_score, acc