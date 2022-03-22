import os
import torch
import random
from operator import itemgetter
from tools.load_data_from_author import re_load


def statistic():
    all_labels_list = [0 for _ in range(8)]
    train_data_list, test_data_list = re_load(0)

    def helper(data_list, flag='train'):
        statistic_dict = dict()
        for paragraph_data in data_list:
            for i in range(len(paragraph_data[0])):
                main_verb = paragraph_data[0][i][1]
                raw_sentence = paragraph_data[0][i][0]
                entity_type = int(paragraph_data[1][i])
                if entity_type == 7:
                    continue
                all_labels_list[entity_type] += 1
                all_labels_list[7] += 1
                if main_verb not in list(statistic_dict.keys()):
                    statistic_dict[main_verb] = [0 for _ in range(8)]
                    statistic_dict[main_verb][entity_type] += 1
                    statistic_dict[main_verb][7] += 1
                else:
                    statistic_dict[main_verb][entity_type] += 1
                    statistic_dict[main_verb][7] += 1

        # print(print())
        # print(statistic_dict)
        all_labels_prob_list = [format(all_labels_list[i] / all_labels_list[7], '.2f') for i in range(7)]

        statistic_dict = sorted(statistic_dict.items(), key=lambda x: x[1][7], reverse=True)

        f = open('output/statistic_main_verb_' + flag + '.txt', 'w')
        f.write("all_labels_list:      " + str(all_labels_list) + '\n')
        f.write("all_labels_prob_list: " + str(all_labels_prob_list) + '\n\n')
        for i in range(len(statistic_dict)):
            main_verb = list(statistic_dict[i])[0]
            nums_list = list(statistic_dict[i])[1]
            prob_list = ['%.2f' % (nums_list[j] / nums_list[7]) for j in range(7)]
            f.write(main_verb + ' ' * (20-len(main_verb)) + str(prob_list) + '                ' + str(nums_list) + '\n')
        #
        statistic_dict_plus = dict()
        for i in range(len(statistic_dict)):
            main_verb = list(statistic_dict[i])[0]
            nums_list = list(statistic_dict[i])[1]
            prob_list = [round(nums_list[j] / nums_list[7], 2) for j in range(7)]
            statistic_dict_plus[main_verb] = [nums_list, prob_list]

        # # entity_type_list = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE']
        # state_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][0], reverse=True)
        # event_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][1], reverse=True)
        # report_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][2], reverse=True)
        # generic_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][3], reverse=True)
        # generalizing_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][4], reverse=True)
        # question_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][5], reverse=True)
        # imperative_sort = sorted(statistic_dict_plus.items(), key=lambda x: x[1][1][6], reverse=True)

        # def do_simple(inputs):
        #     return [[inputs[i][0], inputs[i][1][0][7], inputs[i][1][1][0]] for i in range(len(inputs))]
        #
        # state_sort = do_simple(state_sort)
        # event_sort = do_simple(event_sort)
        # report_sort = do_simple(report_sort)
        # generic_sort = do_simple(generic_sort)
        # generalizing_sort = do_simple(generalizing_sort)
        # question_sort = do_simple(question_sort)
        # imperative_sort = do_simple(imperative_sort)

        torch.save(statistic_dict_plus, 'resource/statistic_dict_plus_' + flag + '.pt')

    helper(train_data_list, 'train')
    helper(test_data_list, 'test')
