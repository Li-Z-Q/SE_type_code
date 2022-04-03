import torch
from tools.print_evaluation_result import print_evaluation_result


def test_model(test_data_list, model):
    model.eval()
    useful_target_Y_list = []
    useful_predict_Y_list = []
    with torch.no_grad():
        for test_data in test_data_list:
            gold_label = test_data[1]
            inputs = test_data[2]  # for BiLSTM is words_embeddings_list, for BERT is words_ids_list

            pre_label, loss = model.forward(inputs, gold_label)  # 1 * 7

            useful_target_Y_list.append(gold_label)
            useful_predict_Y_list.append(pre_label)

    print('\n\ntest print ')

    f1_score, acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)

    return f1_score, acc

