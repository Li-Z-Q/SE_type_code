import torch
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch):

    best_model = None
    best_epoch = None
    best_macro_Fscore = -1
    for epoch in range(total_epoch):
        print('\n\nepoch ' + str(epoch) + '/' + str(total_epoch))

        if hasattr(model, 'reset'):
            model.reset()

        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                gold_label = train_data[1]
                inputs = train_data[2]  # for BiLSTM is words_embeddings_list, for BERT is words_ids_list

                pre_label, loss = model.forward(inputs, gold_label)  # 1 * 7

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                gold_label = valid_data[1]
                inputs = valid_data[2]  # for BiLSTM is words_embeddings_list, for BERT is words_ids_list

                pre_label, loss = model.forward(inputs, gold_label)  # 1 * 7

                useful_target_Y_list.append(gold_label)
                useful_predict_Y_list.append(pre_label)

        # ################################### print and save models ##############################
        tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = model
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore