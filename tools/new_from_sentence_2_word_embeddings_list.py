import nltk
import torch
import gensim
from nltk.tree import Tree

# from stanfordcorenlp import StanfordCoreNLP
# stanford_nlp = StanfordCoreNLP(r'D://stanford-corenlp-4.3.1')  # default english

# n_list = []
# p_list = []

ne_type_list = ['ORGANIZATION', 'LOCATION', 'PERSON', 'MONEY', 'PERCENT', 'DATE', 'TIME']
pos_tag_type_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',  'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
word2vec_vocab = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin', binary=True)


def from_sentence_2_word_embeddings_list(sentence, stanford_nlp):

    # global p_list, n_list
    if '%' in sentence:
        sentence = sentence.replace('%', '%25')
    # print(sentence)
    # # n = nlp.ner(sentence)
    # for p in nlp.pos_tag(sentence):
    #     if p[1] not in p_list and p[0] != p[1] and ord(p[1][0]) >= 65 and ord(p[1][0]) <= 90:
    #         p_list.append(p[1])
    #     print(len(p_list), p_list)
    #
    # for n in nlp.ner(sentence):
    #     if n[1] not in n_list and n[0] != n[1]:
    #         n_list.append(n[1])
    #     print(len(n_list), n_list)

    word_embeddings_list = []
    words_list = stanford_nlp.word_tokenize(sentence)
    word_pos_tags_list = stanford_nlp.pos_tag(sentence)
    word_ners_list = stanford_nlp.ner(sentence)

    # word_ners_list = [7 for _ in range(len(word_pos_tags_list))]
    # for ners in nltk.ne_chunk(word_pos_tags_list):
    #     if type(ners) == Tree:
    #         print(ners)
    #         for word in ners:  # word is ('word', 'pos tag')
    #             print(word)
    #             word_id = words_list.index(word[0])
    #             word_ners_list[word_id] = ne_type_list.index(ners._label)
    # print(word_ners_list)

    for i in range(len(words_list)):
        word = words_list[i]

        if word in word2vec_vocab:
            word_embedding = torch.tensor(word2vec_vocab[word])
        else:
            word_embedding = torch.rand(300) / 2 - 0.25

        word_pos_one_hot = [0 for _ in range(36)]
        if word_pos_tags_list[i] in pos_tag_type_list:
            word_pos_one_hot[pos_tag_type_list.index(word_pos_tags_list[i])] = 1
        word_pos_one_hot = torch.tensor(word_pos_one_hot)
        word_embedding = torch.cat((word_embedding, word_pos_one_hot))

        word_ne_one_hot = [0 for _ in range(7)]
        if word_ners_list[i] in ne_type_list:
            word_ne_one_hot[ne_type_list.index(word_ners_list[i])] = 1
        word_ne_one_hot = torch.tensor(word_ne_one_hot)
        word_embedding = torch.cat((word_embedding, word_ne_one_hot))

        word_embeddings_list.append(word_embedding)
    word_embeddings_list = torch.stack(word_embeddings_list)

    # stanford_nlp.close()

    return word_embeddings_list