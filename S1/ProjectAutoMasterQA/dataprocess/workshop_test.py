

import numpy as np
import pandas as pd
from jieba import posseg
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def segment(sentence, cut_type='word',pos=False):
    if pos:
        word_pos_seq = posseg.lcut(sentence)
        word_seq, pos_seq=[],[]
        if cut_type == 'word':
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq=[]
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


print(segment('我喜欢成都的生活','word'))
print(segment('我喜欢成都的生活','char'))
print(segment('我喜欢成都的生活','word',True))
print(segment('我喜欢成都的生活','char',True))


def train_w2v_model(path):
    w2v_model=Word2Vec(LineSentence(path),workers=4,size=50,min_count=1)
    w2v_model.save('w2v.model')
    # w2v_model.wv.save('name2.model') #速度更快，模型更小，但是加载模型后无法增加新的语料进行训练

def get_model_from_file():
    model = Word2Vec.load('w2v.model')
    return model

if __name__ == '__main__':
    path1 = '../datasets/small_corpus.txt'
    path2 = '../datasets/sentences.txt'
    train_w2v_model(path1)
    model = get_model_from_file()
    print(model.most_similar('车'))

    with open(path2, 'r', encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    new_words=[]
    for line in data:
        line = line.strip().split(' ')
        new_words.append(line)
    model.train(sentences=new_words,epochs=1,total_examples=len(new_words))
    print(model.most_similar('车'))
