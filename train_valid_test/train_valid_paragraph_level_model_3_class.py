import copy
import torch
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
                # raw_sentence_list = train_data[0]
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in train_data[3]]
                _, output, _ = model.forward(inputs)  # sentence_num * 7

                output_2 = output[0]  # 3 * 7
                output_1_list = output[1]  #
                for i in range(len(train_data[1])):
                    gold_label = train_data[1][i]
                    if gold_label == 7:
                        continue
                    if gold_label < 3:
                        new_gold_label = 0
                    elif gold_label < 5:
                        new_gold_label = 1
                    else:
                        new_gold_label = 2
                    batch_loss += -output_2[i][new_gold_label]
                    if ex_loss:
                        batch_loss += -output_1_list[i][new_gold_label]

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        useful_predict_Y_1_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in valid_data[3]]
                pre_labels_list, _, _ = model.forward(inputs)  # sentence_num * 7
                pre_labels_list_2 = pre_labels_list[0]

                gold_labels_list = [int(gold_label) for gold_label in valid_data[1] if gold_label != 7]
                mapping_dict = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
                useful_target_Y_list += [mapping_dict[g] for g in gold_labels_list]
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

    return best_epoch, best_model, best_macro_f1, best_acc
