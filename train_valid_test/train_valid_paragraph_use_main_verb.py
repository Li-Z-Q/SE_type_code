import copy
import numpy
import torch
import torch.nn.functional as F
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid_fn(model, optimizer, train_batch_list, valid_data_list, total_epoch):
    best_acc = 0
    best_model = None
    best_epoch = None
    best_macro_f1 = -1

    for epoch in range(total_epoch):
        print('\nepoch ' + str(epoch) + '/' + str(total_epoch))
        # print("before train: ", torch.cuda.memory_allocated(0))

        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                main_verb_list = [train_data[0][i][1] for i in range(len(train_data[0]))]
                main_verb_embedding_list = [train_data[0][i][2] for i in range(len(train_data[0]))]
                main_verb_position_list = [train_data[0][i][3] for i in range(len(train_data[0]))]
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in train_data[3]]
                _, [output_2, _], _ = model.forward(inputs, main_verb_list, main_verb_position_list)  # sentence_num * 7

                for i in range(len(train_data[1])):
                    gold_label = train_data[1][i]
                    if gold_label == 7:
                        continue
                    batch_loss += -output_2[i][gold_label]
                    # if main_verb_list[i] in list(model.main_verb_contribution.keys()):
                    #     if model.main_verb_contribution[main_verb_list[i]][0][7] > 5:
                    #         main_verb_contribution = model.main_verb_contribution[main_verb_list[i]][1]  # a list, len is 7
                    #         main_verb_contribution = torch.tensor(numpy.array(main_verb_contribution)).to(torch.float32).cuda()
                    #         kl = F.kl_div(output_2[i].softmax(dim=-1).log(), main_verb_contribution.softmax(dim=-1), reduction='sum')
                    #         batch_loss += kl

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                main_verb_list = [valid_data[0][i][1] for i in range(len(valid_data[0]))]
                main_verb_embedding_list = [valid_data[0][i][2] for i in range(len(valid_data[0]))]
                main_verb_position_list = [valid_data[0][i][3] for i in range(len(valid_data[0]))]
                inputs = [torch.tensor(sentence)[:, :].cuda() for sentence in valid_data[3]]

                pre_labels_list, _, _ = model.forward(inputs, main_verb_list, main_verb_position_list)  # sentence_num * 7
                pre_labels_list_2 = pre_labels_list[0]

                useful_target_Y_list += [int(gold_label) for gold_label in valid_data[1] if gold_label != 7]
                useful_predict_Y_list += [pre_labels_list_2[i] for i in range(len(valid_data[1])) if valid_data[1][i] != 7]

        # ################################### print and save models ##############################
        tmp_macro_f1, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_f1 > best_macro_f1:
            best_acc = tmp_acc
            best_epoch = epoch
            best_macro_f1 = tmp_macro_f1
            best_model = copy.deepcopy(model)

    return best_epoch, best_model, best_macro_f1, best_acc
