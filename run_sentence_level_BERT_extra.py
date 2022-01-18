import random

import torch
from torch import optim
from models.sentence_level_BERT_extra import MyModel
from tools.devide_train_batch import get_train_batch_list
from train_valid_test.train_valid_sentence_level_model import train_and_valid_extra


def from_paragraph_to_sentence(paragraph_data_list):
    sentence_data_list = []

    for paragraph_data in paragraph_data_list:
        for sentence, label in zip(paragraph_data[3], paragraph_data[1]):
            # for BERT, sentence is words_embeddings_list
            if label != 7:
                sentence_data_list.append(["LZQ", label, sentence])

    random.shuffle(sentence_data_list)

    return sentence_data_list


def main(paragraph_train_data_list, paragraph_valid_data_list, paragraph_test_data_list):
    print("\n\nstart sentence level BERT  extra")

    EPOCHs = 4
    DROPOUT = 0.5
    BATCH_SIZE = 4
    LEARN_RATE = 1e-5
    WEIGHT_DECAY = 1e-4

    test_data_list = from_paragraph_to_sentence(paragraph_test_data_list)
    valid_data_list = from_paragraph_to_sentence(paragraph_valid_data_list)
    train_data_list = from_paragraph_to_sentence(paragraph_train_data_list)

    train_batch_list = get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len=1)

    model = MyModel(dropout=DROPOUT).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

    best_epoch, best_model, best_macro_Fscore, best_acc = train_and_valid_extra(model, optimizer, train_batch_list,
                                                                          valid_data_list, EPOCHs)
    torch.save(best_model, 'models/model_sentence_level_BERT_extra.pt')
    print("sentence level BERT  extra best_epoch: ", best_epoch, best_macro_Fscore, best_acc)

    # test_model(test_data_list, best_model)

    print("sentence level BERT  extra end\n\n********************************************")