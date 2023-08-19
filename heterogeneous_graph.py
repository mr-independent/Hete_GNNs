# -*- coding: utf-8 -*-
import numpy as np
import spacy
import pickle
import nltk

nlp = spacy.load('en_core_web_sm')

def heterogeneous_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len_1 = len(text.split())
    pos = [token.pos_ for token in document]
    seq_len = (len(pos))

    matrix = np.zeros((seq_len + 5, seq_len + 5)).astype('float32')

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    # m = ['ADJ', 'NOUN', 'ADP']
    m = ['ADJ',  'ADP']
    f1 = open('./datasets/negative-words.txt', 'r',  encoding='gb18030')
    f2 = open('./datasets/positive-words.txt','r', encoding='gb18030')
    negative_dic = [line.split()[0] for line in f1.readlines()]
    positive_dic = [line.split()[0] for line in f2.readlines()]
    voc = text.split()
    x = []
    i = 0
    xtend_pos = ['ADJ', 'ADP', 'pos', 'neg', 'neu']
    pos.extend(xtend_pos)
    voc.extend(xtend_pos)
    for i in range(len(pos)):
        matrix[i][i] = 1
        if pos[i] == m[0]:
            matrix[i][-5] = 1
            matrix[-5][i] = 1
        elif pos[i] == m[1]:
            matrix[i][-4] = 1
            matrix[-4][i] = 1

    for i in range(len(voc)):
        matrix[i][i] = 1
        if voc[i] in negative_dic:
            matrix[i][-2] =1
            matrix[-2][i] = 1
        elif voc[i] in positive_dic:
            matrix[i][-3] = 1
            matrix[-3][i] = 1

    return matrix


def heterogeneous_adj_matrix_use_nltk(text):
    document = nlp(text)
    seq_len = len(text.split())

    matrix = np.zeros((seq_len + 5, seq_len + 5)).astype('float32')

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    doc = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(doc)
    p = [pos[1] for pos in pos_tag]
    voc = text.split()

    new_pos = []
    if seq_len == len(p):
        new_pos = p
    else:
        j = 0
        for i in range(seq_len):
            if j != len(p) and voc[i] == pos_tag[j][0]:
                new_pos.append(pos_tag[j][1])
                j += 1
            else:
                new_pos.append('')
                if j + 1 != len(p) and i + 1 != seq_len:  # Judge whether it is out of bounds
                    temp = j + 1
                    while temp != len(p) and voc[i+1] != pos_tag[temp][0]:  # Find the position of the next word
                        temp += 1
                    if temp == len(p):
                        j += 1
                    else:
                        j = temp

    adj = ['JJ', 'JJR', 'JJS']
    adv = ['RB', 'RBR', 'RBS']
    f1 = open('./datasets/negative-words.txt', 'r', encoding='gb18030')
    f2 = open('./datasets/positive-words.txt', 'r', encoding='gb18030')
    negative_dic = [line.split()[0] for line in f1.readlines()]
    positive_dic = [line.split()[0] for line in f2.readlines()]

    x = []
    i = 0
    xtend_pos = ['ADJ', 'ADV', 'pos', 'neg', 'neu']
    new_pos.extend(xtend_pos)
    voc.extend(xtend_pos)

    for i in range(len(new_pos)):
        matrix[i][i] = 1
        if new_pos[i] in adj:
            matrix[i][-5] = 1
            matrix[-5][i] = 1
        elif new_pos[i] in adv:
            matrix[i][-4] = 1
            matrix[-4][i] = 1

    for i in range(len(voc)):
        matrix[i][i] = 1
        if voc[i] in negative_dic:
            matrix[i][-2] = 1
            matrix[-2][i] = 1
        elif voc[i] in positive_dic:
            matrix[i][-3] = 1
            matrix[-3][i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        # adj_matrix = heterogeneous_adj_matrix(text_left+' '+aspect+' '+text_right)
        adj_matrix = heterogeneous_adj_matrix_use_nltk(text_left + ' ' + aspect + ' ' + text_right)
        # test
        if adj_matrix is not None:
            idx2graph[i] = adj_matrix
        # idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()
    print("process is end ====", filename)

if __name__ == '__main__':

    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')