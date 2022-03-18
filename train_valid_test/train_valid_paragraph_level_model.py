import copy
import torch
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch):
    best_acc = 0
    best_model = None
    best_epoch = None
    best_macro_f1 = -1

    write_time = 0
    for epoch in range(total_epoch):
        print('\n\nepoch ' + str(epoch) + '/' + str(total_epoch))
        # print("before train: ", torch.cuda.memory_allocated(0))

        # ################################### train ##############################
        model.train()
        if hasattr(model, 'valid_flag'):
            model.valid_flag = 0
            model.display_first()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                # raw_sentence_list = train_data[0]
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in train_data[3]]
                _, loss, _ = model.forward(inputs, train_data[1])  # sentence_num * 7
                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        if hasattr(model, 'valid_flag'):
            model.valid_flag = 1

        useful_target_Y_list = []
        useful_predict_Y_list = []
        useful_raw_sentence_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in valid_data[3]]
                pre_labels_list, _, _ = model.forward(inputs, valid_data[1])  # sentence_num * 7
                gold_labels_list = [int(gold_label) for gold_label in valid_data[1] if gold_label != 7]
                raw_sentence_list = [valid_data[0][i] for i in range(len(valid_data[1])) if valid_data[1][i] != 7]

                useful_predict_Y_list += pre_labels_list
                useful_target_Y_list += gold_labels_list
                useful_raw_sentence_list += raw_sentence_list

        # ################################### print and save models ##############################
        tmp_macro_f1, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_f1 > best_macro_f1:
            best_acc = tmp_acc
            best_epoch = epoch
            best_macro_f1 = tmp_macro_f1
            best_model = copy.deepcopy(model)

        # if tmp_macro_f1 < 0.62 or tmp_macro_f1 > 0.72:
        #     for i in range(len(useful_target_Y_list)):
        #         pre_label = useful_predict_Y_list[i]
        #         gold_label = useful_target_Y_list[i]
        #         with open('output/paragraph_statistics/' + str(epoch) + '_gold_' + str(gold_label) + '_pre_' + str(pre_label) + '.txt', mode='a') as f:
        #             f.write(useful_raw_sentence_list[i][1] + '\n')  # main_verb
        #             f.write(useful_raw_sentence_list[i][0] + '\n\n')  # raw_sentence
        #     write_time += 1
        # if write_time == 2:
        #     break

    return best_epoch, best_model, best_macro_f1, best_acc
