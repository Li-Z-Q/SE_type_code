import random

def from_paragraph_to_sentence_fn(paragraph_data_list, random_seed):
    print("from paragraph to sentence")
    data_list = []

    for data in paragraph_data_list:
        for raw_sentence, label, word_embeddings_list in zip(data[0], data[1], data[3]):
            data_list.append([raw_sentence, label, word_embeddings_list])

    print("len(data_list): ", len(data_list))

    random.seed(random_seed)
    random.shuffle(data_list)

    return data_list
