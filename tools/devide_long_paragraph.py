def devide_long_paragraph_to_short(long_data, all_data_list):
    # long_data is [segment_list, label_to_num_list, label_list_len, segment_embeddings_list]

    for i in range(int(long_data[2] / 20)):

        short_data = [long_data[0][i*20:(i+1)*20],
                      long_data[1][i*20:(i+1)*20],
                      len(long_data[0][i*20:(i+1)*20]),
                      long_data[3][i*20:(i+1)*20]]

        all_data_list.append(short_data)