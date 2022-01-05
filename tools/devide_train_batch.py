import random


def get_train_batch_list(train_data_list, BATCH_SIZE, each_data_len):
    random.seed(1)
    random.shuffle(train_data_list)

    train_data_batch_list = [[]]
    total_segment_num_in_this_batch = 0
    for train_data in train_data_list:  # train_data: [X, y]
        if total_segment_num_in_this_batch >= BATCH_SIZE:
            train_data_batch_list.append([])
            total_segment_num_in_this_batch = 0
        train_data_batch_list[-1].append(train_data)
        if each_data_len == 0:  # paragraph-level
            total_segment_num_in_this_batch += train_data[2]  # train_data[2] is valid label num
        else:  # sentence-level
            total_segment_num_in_this_batch += each_data_len

    return train_data_batch_list