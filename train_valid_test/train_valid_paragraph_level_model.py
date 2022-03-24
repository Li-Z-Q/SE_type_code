import copy
import torch
from sklearn import metrics
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch, ex_loss):
    best_acc = 0
    best_model = None
    best_epoch = None
    best_macro_f1 = -1

    write_time = 0
    for epoch in range(total_epoch):
        print('\nepoch ' + str(epoch) + '/' + str(total_epoch))
        # print("before train: ", torch.cuda.memory_allocated(0))

        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in train_data[3]]
                _, output, _ = model.forward(inputs)  # sentence_num * 7

                output_2 = output[0]  # 3 * 7
                output_1_list = output[1]  #
                for i in range(len(train_data[1])):
                    gold_label = train_data[1][i]
                    if gold_label == 7:
                        continue
                    batch_loss += -output_2[i][gold_label]
                    if ex_loss:
                        batch_loss += -output_1_list[i][gold_label]

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        useful_predict_Y_1_list = []
        useful_extra_info_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in valid_data[3]]
                pre_labels_list, _, _ = model.forward(inputs)  # sentence_num * 7
                pre_labels_list_2 = pre_labels_list[0]

                useful_target_Y_list += [int(gold_label) for gold_label in valid_data[1] if gold_label != 7]
                useful_extra_info_list += [valid_data[0][i] for i in range(len(valid_data[1])) if valid_data[1][i] != 7]
                useful_predict_Y_list += [pre_labels_list_2[i] for i in range(len(valid_data[1])) if valid_data[1][i] != 7]

                ex_pre_label_list = pre_labels_list[1]
                if len(ex_pre_label_list) > 0:  # use ex_initial
                    useful_predict_Y_1_list += [ex_pre_label_list[i] for i in range(len(valid_data[1])) if valid_data[1][i] != 7]

        # ################################### print and save models ##############################
        if len(useful_predict_Y_1_list) > 0:
            print('ex_result')
            _, _ = print_evaluation_result(useful_target_Y_list, useful_predict_Y_1_list)
        tmp_macro_f1, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_f1 > best_macro_f1:
            best_acc = tmp_acc
            best_epoch = epoch
            best_macro_f1 = tmp_macro_f1
            best_model = copy.deepcopy(model)

        # entity_type_list = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']
        # if tmp_macro_f1 < 0.62 or tmp_macro_f1 > 0.75:
        #     for i in range(len(useful_target_Y_list)):
        #         pre_label = useful_predict_Y_list[i]
        #         gold_label = useful_target_Y_list[i]
        #         with open('output/paragraph_statistics/' + str(tmp_macro_f1) + '_gold_' + entity_type_list[gold_label] + '_pre_' + entity_type_list[pre_label] + '.txt', mode='a') as f:
        #             f.write(useful_extra_info_list[i][1] + '\n')  # main_verb
        #             f.write(useful_extra_info_list[i][0] + '\n\n')  # raw_sentence
        #     write_time += 1
        # if write_time == 2:
        #     break
        #
        # if tmp_macro_f1 > 0.75:
        #     test_dict_plus = torch.load('resource/statistic_dict_plus_test.pt')
        #     confusion_verb_list = [[dict() for _ in range(7)] for _ in range(7)]
        #     for i in range(len(useful_target_Y_list)):
        #         main_verb = useful_extra_info_list[i][1]
        #         pre_label = useful_predict_Y_list[i]
        #         gold_label = useful_target_Y_list[i]
        #
        #         if main_verb not in list(confusion_verb_list[gold_label][pre_label].keys()):
        #             confusion_verb_list[gold_label][pre_label][main_verb] = 0
        #         confusion_verb_list[gold_label][pre_label][main_verb] += 1
        #
        #         if len(test_dict_plus[main_verb]) < 3:
        #             test_dict_plus[main_verb].append([])  # pre
        #             test_dict_plus[main_verb].append([])  # gold
        #         test_dict_plus[main_verb][2].append(pre_label)
        #         test_dict_plus[main_verb][3].append(gold_label)
        #
        #     f = open('output/paragraph_statistic_main_verb_test_pre_gold_confusion_matrix.txt', 'w')
        #     all_confusion_matrix = metrics.confusion_matrix(useful_target_Y_list, useful_predict_Y_list)
        #     f.write(str(all_confusion_matrix) + '\n')
        #     for i in range(7):
        #         for j in range(7):
        #             l = sorted(confusion_verb_list[i][j].items(), key=lambda x: x[1], reverse=True)
        #             f.write(str(i) + '_be_pre_as_' + str(j) + ': ' + str(l) + '\n')
        #     f.write('\n\n\n\n')
        #
        #     for i in range(len(list(test_dict_plus.keys()))):
        #         main_verb = list(test_dict_plus.keys())[i]
        #         test_dict_plus[main_verb][2] += [0, 1, 2, 3, 4, 5, 6]
        #         test_dict_plus[main_verb][3] += [0, 1, 2, 3, 4, 5, 6]  # to keep 7 * 7
        #         confusion_matrix = metrics.confusion_matrix(test_dict_plus[main_verb][3], test_dict_plus[main_verb][2])
        #         f.write(main_verb + ': -----------------: ' + str(test_dict_plus[main_verb][0]) + '\n')
        #         f.write(str(confusion_matrix) + '\n\n')
        #     input()

    return best_epoch, best_model, best_macro_f1, best_acc
