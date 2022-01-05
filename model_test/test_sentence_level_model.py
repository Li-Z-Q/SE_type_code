import torch



def test_model(test_data_list, model):
    model.eval()
    useful_target_Y_list = []
    useful_predict_Y_list = []
    with torch.no_grad():
        for valid_data in test_data_list:
            gold_label = valid_data[1]
            sentence = valid_data[0]  # raw sentence
            words_ids_list = tokenizer(sentence, return_tensors="pt").input_ids.cuda()

            pre_label, loss = model.forward(words_ids_list, gold_label)  # 1 * 7

            useful_target_Y_list.append(gold_label)
            useful_predict_Y_list.append(pre_label)

    # ################################### print and save models ##############################
    tmp_macro_Fscore = print_evaluation_result(useful_target_Y_list, useful_predict_Y_list)
