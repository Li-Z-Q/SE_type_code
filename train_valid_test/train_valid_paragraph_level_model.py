import torch
from tools.print_evaluation_result import print_evaluation_result


def train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch):

    best_model = None
    best_epoch = None
    best_macro_Fscore = -1
    for epoch in range(total_epoch):
        print('\n\nepoch ' + str(epoch) + '/' + str(total_epoch))

        # print("before train: ", torch.cuda.memory_allocated(0))
        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                sentences_list = []
                gold_labels_list = []
                for sentence, label in zip(train_data[3], train_data[1]):
                    # for BiLSTM, sentence is words_embeddings_list
                    # for BERT, sentence is words_ids_list
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)

                _, loss = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                sentences_list = []
                gold_labels_list = []
                for sentence, label in zip(valid_data[3], valid_data[1]):
                    # for BiLSTM, sentence is words_embeddings_list
                    # for BERT, sentence is words_ids_list
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)

                pre_labels_list, _ = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                for i in range(len(gold_labels_list)):
                    useful_target_Y_list.append(gold_labels_list[i])
                    useful_predict_Y_list.append(pre_labels_list[i])

        # ################################### print and save models ##############################
        tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = model
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore