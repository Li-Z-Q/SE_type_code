import nltk
import torch
import gensim
from nltk.tree import Tree


print("****************************************** 300 ********************************************************************************")

def from_sentence_2_word_embeddings_list(sentence, stanford_nlp, word2vec_vocab):
    if '%' in sentence:
        sentence = sentence.replace('%', '%25')

    words_list = stanford_nlp.word_tokenize(sentence)

    ################################################################### 343 ##################################################
    # # ne_type_list = ['ORGANIZATION', 'LOCATION', 'PERSON', 'MONEY', 'PERCENT', 'DATE', 'TIME']
    # ne_type_list = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC', 'MONEY', 'NUMBER',
    #                 'ORDINAL', 'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET',
    #                 'EMAIL', 'URL', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'NATIONALITY',
    #                 'RELIGION', 'TITLE', 'IDEOLOGY', 'CRIMINAL_CHARGE', 'CAUSE_OF_DEATH']
    # pos_tag_type_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    #                      'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
    #                      'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    n_list = stanford_nlp.ner(sentence)
    p_list = stanford_nlp.pos_tag(sentence)
    #############################################################################################################################

    word_embeddings_list = []
    for i in range(len(words_list)):
        word = words_list[i]

        if word in word2vec_vocab:
            word_embedding = torch.tensor(word2vec_vocab[word])
        else:
            word_embedding = torch.rand(300) / 2 - 0.25

        ################################################################### 343 ##################################################
        # word_pos_one_hot = [0 for _ in range(len(pos_tag_type_list))]
        # p = p_list[i][1]
        # if p in pos_tag_type_list:
        #     word_pos_one_hot[pos_tag_type_list.index(p)] = 1
        # word_pos_one_hot = torch.tensor(word_pos_one_hot)
        # word_embedding = torch.cat((word_embedding, word_pos_one_hot), dim=0)

        # ne_one_hot = [0 for _ in range(len(ne_type_list))]
        # n = n_list[i][1]
        # if n in ne_type_list:
        #     ne_one_hot[ne_type_list.index(n)] = 1
        # ne_one_hot = torch.tensor(ne_one_hot)
        # word_embedding = torch.cat((word_embedding, ne_one_hot), dim=0)
        #####################################################################################################################

        word_embedding = word_embedding.tolist()
        word_embeddings_list.append(word_embedding)  # is a list for writing to json file

    return word_embeddings_list
