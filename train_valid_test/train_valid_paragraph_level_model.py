import copy
import torch
from tools.print_evaluation_result import print_evaluation_result
from models.paragraph_level_BiLSTM_label_embedding_MLP_pre import MyModel


def train_and_valid(model, optimizer, train_batch_list, valid_data_list, total_epoch, with_raw_text=False):
    best_acc = 0
    best_model = None
    best_epoch = None
    best_macro_Fscore = -1
    for epoch in range(total_epoch):
        print('\n\nepoch ' + str(epoch) + '/' + str(total_epoch))

        if hasattr(model, 'reset'):
            print("model.reset")
            model.reset()

        # print("before train: ", torch.cuda.memory_allocated(0))
        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                sentences_list = []
                gold_labels_list = []
                raw_sentences_list = []
                for sentence, label, raw_sentence in zip(train_data[3], train_data[1], train_data[0]):
                    # for BiLSTM, sentence is words_embeddings_list
                    # for BERT, sreentence is words_ids_list
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)
                        raw_sentences_list.append(raw_sentence)

                if with_raw_text:
                    pre_labels_list, loss = model.forward([sentences_list, raw_sentences_list], gold_labels_list)  # sentence_num * 7

                else:
                    pre_labels_list, loss = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                # print("train pre : ", pre_labels_list)
                # print("train gold: ", gold_labels_list)

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        if hasattr(model, 'valid_flag'):
            model.valid_flag = True
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                sentences_list = []
                gold_labels_list = []
                raw_sentences_list = []
                for sentence, label, raw_sentence in zip(valid_data[3], valid_data[1], valid_data[0]):
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

                # print("valid pre : ", pre_labels_list)
                # print("valid gold: ", gold_labels_list)

                try:
                    for i in range(len(pre_labels_list)):
                        useful_target_Y_list.append(gold_labels_list[i])
                        useful_predict_Y_list.append(pre_labels_list[i])
                except:
                    print("gold_labels_list: ", gold_labels_list)
                    print("pre_labels_list:  ", pre_labels_list)

        # ################################### print and save models ##############################
        tmp_macro_Fscore, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_acc = tmp_acc
            best_macro_Fscore = tmp_macro_Fscore

    return best_epoch, best_model, best_macro_Fscore, best_acc


def train_and_valid_special_deepcopy(model, temp_best_model, optimizer, train_batch_list, valid_data_list, total_epoch, with_raw_text=False):
    best_acc = 0
    best_epoch = None
    best_macro_Fscore = -1
    best_model = temp_best_model
    for epoch in range(total_epoch):
        print('\n\nepoch ' + str(epoch) + '/' + str(total_epoch))

        if hasattr(model, 'reset'):
            print("model.reset")
            model.reset()

        # print("before train: ", torch.cuda.memory_allocated(0))
        # ################################### train ##############################
        model.train()
        for train_batch in train_batch_list:
            batch_loss = 0
            optimizer.zero_grad()
            for train_data in train_batch:
                sentences_list = []
                gold_labels_list = []
                raw_sentences_list = []
                for sentence, label, raw_sentence in zip(train_data[3], train_data[1], train_data[0]):
                    # for BiLSTM, sentence is words_embeddings_list
                    # for BERT, sreentence is words_ids_list
                    if label != 7:
                        gold_labels_list.append(label)
                        sentences_list.append(sentence)
                        raw_sentences_list.append(raw_sentence)

                if with_raw_text:
                    pre_labels_list, loss = model.forward([sentences_list, raw_sentences_list], gold_labels_list)  # sentence_num * 7

                else:
                    pre_labels_list, loss = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7

                # print("train pre : ", pre_labels_list)
                # print("train gold: ", gold_labels_list)

                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        # ################################### valid ##############################
        if hasattr(model, 'valid_flag'):
            model.valid_flag = True
        model.eval()
        useful_target_Y_list = []
        useful_predict_Y_list = []
        with torch.no_grad():
            for valid_data in valid_data_list:
                sentences_list = []
                gold_labels_list = []
                raw_sentences_list = []
                for sentence, label, raw_sentence in zip(valid_data[3], valid_data[1], valid_data[0]):
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

                # print("valid pre : ", pre_labels_list)
                # print("valid gold: ", gold_labels_list)

                try:
                    for i in range(len(pre_labels_list)):
                        useful_target_Y_list.append(gold_labels_list[i])
                        useful_predict_Y_list.append(pre_labels_list[i])
                except:
                    print("gold_labels_list: ", gold_labels_list)
                    print("pre_labels_list:  ", pre_labels_list)

        # ################################### print and save models ##############################
        tmp_macro_Fscore, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        if tmp_macro_Fscore > best_macro_Fscore:
            best_epoch = epoch
            best_acc = tmp_acc
            best_macro_Fscore = tmp_macro_Fscore
            best_model.load_state_dict(model.state_dict())  # aim to get deepcopy things

        # # check load_state_dict whether do deepcopy  #####################################################################
        # print('\nmodel in valid')
        # model.eval()
        # useful_target_Y_list = []
        # useful_predict_Y_list = []
        # with torch.no_grad():
        #     for valid_data in valid_data_list:
        #         sentences_list = []
        #         gold_labels_list = []
        #         raw_sentences_list = []
        #         for sentence, label, raw_sentence in zip(valid_data[3], valid_data[1], valid_data[0]):
        #             if label != 7:
        #                 gold_labels_list.append(label)
        #                 sentences_list.append(sentence)
        #                 raw_sentences_list.append(raw_sentence)
        #
        #         if with_raw_text:
        #             pre_labels_list, _ = model.forward([sentences_list, raw_sentences_list], gold_labels_list)  # sentence_num * 7
        #         else:
        #             pre_labels_list, _ = model.forward(sentences_list, gold_labels_list)  # sentence_num * 7
        #
        #         for i in range(len(pre_labels_list)):
        #             useful_target_Y_list.append(gold_labels_list[i])
        #             useful_predict_Y_list.append(pre_labels_list[i])
        # tmp_macro_Fscore, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        #
        # print('\ntemp_best_model in valid')
        # best_model.eval()
        # useful_target_Y_list = []
        # useful_predict_Y_list = []
        # with torch.no_grad():
        #     for valid_data in valid_data_list:
        #         sentences_list = []
        #         gold_labels_list = []
        #         raw_sentences_list = []
        #         for sentence, label, raw_sentence in zip(valid_data[3], valid_data[1], valid_data[0]):
        #             if label != 7:
        #                 gold_labels_list.append(label)
        #                 sentences_list.append(sentence)
        #                 raw_sentences_list.append(raw_sentence)
        #
        #         if with_raw_text:
        #             pre_labels_list, _ = best_model.forward([sentences_list, raw_sentences_list], gold_labels_list)  # sentence_num * 7
        #         else:
        #             pre_labels_list, _ = best_model.forward(sentences_list, gold_labels_list)  # sentence_num * 7
        #
        #         for i in range(len(pre_labels_list)):
        #             useful_target_Y_list.append(gold_labels_list[i])
        #             useful_predict_Y_list.append(pre_labels_list[i])
        # tmp_macro_Fscore, tmp_acc = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
        # # check load_state_dict whether do deepcopy  #####################################################################

    return best_epoch, best_model, best_macro_Fscore, best_acc
