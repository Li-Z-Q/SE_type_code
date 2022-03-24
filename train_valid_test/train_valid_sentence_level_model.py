import copy
import torch
from sklearn import metrics
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid_fn(model, optimizer, train_batch_list, valid_data_list, total_epoch):

    best_acc = -1
    best_model = None
    best_epoch = None
    best_macro_Fscore = -1
    for epoch in range(total_epoch):
        print('\nepoch ' + str(epoch) + '/' + str(total_epoch))

        write_time = 0
        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                gold_label = train_data[1]
                inputs = torch.tensor(train_data[2]).cuda()  # for BiLSTM is words_embeddings_list, for BERT is words_ids_list

                pre_label, output, _ = model.forward(inputs)  # 1 * 7
                if gold_label != 7:
                    batch_loss += -output[gold_label]

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        useful_extra_info_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                gold_label = valid_data[1]
                inputs = torch.tensor(valid_data[2]).cuda()  # for BiLSTM is words_embeddings_list, for BERT is words_ids_list

                pre_label, _, _ = model.forward(inputs)  # 1 * 7
                if gold_label != 7:
                    useful_target_Y_list.append(gold_label)
                    useful_predict_Y_list.append(pre_label)
                    useful_extra_info_list.append(valid_data[0])

        # ################################### print and save models ##############################
        tmp_macro_f1, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_f1 > best_macro_Fscore:
            best_acc = tmp_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_macro_Fscore = tmp_macro_f1

        # do statistic
        entity_type_list = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']
        if tmp_macro_f1 < 0.65 or tmp_macro_f1 > 0.7:
            for i in range(len(useful_target_Y_list)):
                pre_label = useful_predict_Y_list[i]
                gold_label = useful_target_Y_list[i]
                with open('output/sentence_statistic/' + str(tmp_macro_f1) + '_gold_' + entity_type_list[gold_label] + '_pre_' + entity_type_list[pre_label] + '.txt', mode='a') as f:
                    f.write(useful_extra_info_list[i][1] + '\n')  # main_verb
                    f.write(useful_extra_info_list[i][0] + '\n\n')  # raw_sentence
            write_time += 1
        if write_time == 2:
            break

        if tmp_macro_f1 > 0.7:
            test_dict_plus = torch.load('resource/statistic_dict_plus_test.pt')
            confusion_verb_list = [[dict() for _ in range(7)] for _ in range(7)]
            for i in range(len(useful_target_Y_list)):
                main_verb = useful_extra_info_list[i][1]
                pre_label = useful_predict_Y_list[i]
                gold_label = useful_target_Y_list[i]

                if main_verb not in list(confusion_verb_list[gold_label][pre_label].keys()):
                    confusion_verb_list[gold_label][pre_label][main_verb] = 0
                confusion_verb_list[gold_label][pre_label][main_verb] += 1

                if len(test_dict_plus[main_verb]) < 3:
                    test_dict_plus[main_verb].append([])  # pre
                    test_dict_plus[main_verb].append([])  # gold
                test_dict_plus[main_verb][2].append(pre_label)
                test_dict_plus[main_verb][3].append(gold_label)

            f = open('output/sentence_statistic_main_verb_test_pre_gold_confusion_matrix.txt', 'w')
            all_confusion_matrix = metrics.confusion_matrix(useful_target_Y_list, useful_predict_Y_list)
            f.write(str(all_confusion_matrix) + '\n')
            for i in range(7):
                for j in range(7):
                    l = sorted(confusion_verb_list[i][j].items(), key=lambda x: x[1], reverse=True)
                    f.write(str(i) + '_be_pre_as_' + str(j) + ': ' + str(l) + '\n')
            f.write('\n\n\n\n')

            for i in range(len(list(test_dict_plus.keys()))):
                main_verb = list(test_dict_plus.keys())[i]
                test_dict_plus[main_verb][2] += [0, 1, 2, 3, 4, 5, 6]
                test_dict_plus[main_verb][3] += [0, 1, 2, 3, 4, 5, 6]  # to keep 7 * 7
                confusion_matrix = metrics.confusion_matrix(test_dict_plus[main_verb][3], test_dict_plus[main_verb][2])
                f.write(main_verb + ': -----------------: ' + str(test_dict_plus[main_verb][0]) + '\n')
                f.write(str(confusion_matrix) + '\n\n')
            input()

    return best_epoch, best_model, best_macro_Fscore, best_acc