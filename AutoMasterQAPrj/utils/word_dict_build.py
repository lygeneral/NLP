# L1(step2): 根据分词建立词典
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_data(path):
    """
    @description: 读取分词数据
    @param: path-分词数据路径
    @return: word-分词list
    """
    with open(path, 'r', encoding='utf-8') as f:
        word = []
        for line in f:
            word += line.split()
    print('%s data reading is finished' % path)
    return word

def save_word_dict(voacb_path, word_item):
    """
    @description: 将分词字典写入文件中
    @param: voacb_path-字典路径, word_item-分词字典[(xx,xx),(xx,xx)]
    @return: None
    """
    with open(voacb_path, 'w', encoding='utf-8') as f:
        for line in word_item:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            print(item)
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab

def vocab_build(voacb_path, train_segx_path, train_segy_path, test_segx_path):
    """
    @description: 读取训练集和测试集分词形成分词字典库，按词频由高到低排列
    @param: voacb_path-字典路径, train_segx_path-分词后的训练集x路径, train_segy_path-分词后的训练集y路径, test_segx_path-分词后的测试集x路径
    @return: None
    """
    word_list = []
    word_list += read_data(train_segx_path)
    word_list += read_data(train_segy_path)
    word_list += read_data(test_segx_path)

    vocab, reverse_vocab = build_vocab(word_list)
    save_word_dict(voacb_path, vocab)

if __name__ == '__main__':
    train_segx_path = '{}/datasets/train_set.seg_x.txt'.format(BASE_DIR)
    train_segy_path = '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR)
    test_segx_path = '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR)
    voacb_path = '{}/datasets/vocab.txt'.format(BASE_DIR)

    # 由训练集和测试集分词建立字典
    vocab_build(voacb_path, train_segx_path, train_segy_path, test_segx_path)