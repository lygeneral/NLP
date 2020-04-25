import jieba
import pandas as pd
from collections import Counter

def read_stopword(path):
    """
    @description: 从文件中读取停用词并返回set集合
    @input: path-停用词路径
    @return: lines-停用词set集合
    """
    lines = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

def remove_word(word_list, remove_words):
    """
    @description: 去除list中的无效词、字符
    @input: word_list-原始list
    @return: word_list-去除无效词、字符后的list
    """
    word_list = [word for word in word_list if word not in remove_words]
    return word_list

def parse_data(train_path, test_path):
    """
    @description: 读取训练数据和测试数据
    @input: train_path-训练数据路径，test_path-测试数据路径
    @return: train_x-训练数据x, train_y-训练数据y, test_x-测试数据x, test_y-测试数据y（无数据）
    """
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'],how='any',inplace=True)
    # null填充为空字符
    train_df.fillna('', inplace=True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = train_df.Report
    assert len(train_x) == len(train_y)

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []

    return train_x, train_y, test_x, test_y


def segment(sentence):
    """
    @description: jieba分词
    @input: 字符串
    @return: list
    """
    return jieba.lcut(sentence)

def save_data(data, path, remove_words):
    """
    @description: 输入数据并在去除停用词及无效词后进行分词
    @input: data-数据集, path-写入文件的路径, remove_words-待去除的无效词
    @return: None
    """
    with open(path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data:
            if isinstance(line, str):
                seg_list = segment(line.strip())
                test = seg_list
                seg_list = remove_word(seg_list, remove_words)
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f.write('%s' % seg_line)
                    f.write('\n')
                    count += 1
                else:
                    f.write('随时 联系')
                    f.write('\n')
                    count += 1
    print('%s is finished, length is ' % path, count)

def wordvec_build(train_data_path, test_data_path, train_segx_path, train_segy_path, test_segx_path, remove_words):
    """
    @description: 将训练集和测试集的数据进行分词
    @input: train_data_path-数据集路径, test_data_path-测试集路径, remove_words-待去除的无效词
    @input: train_segx_path-分词后的训练集x路径, train_segy_path-分词后的训练集y路径, test_segx_path-分词后的测试集x路径
    @return: None
    """
    train_x, train_y, test_x, _ = parse_data(train_data_path, test_data_path)
    save_data(train_x, train_segx_path, remove_words)
    save_data(train_y, train_segy_path, remove_words)
    save_data(test_x, test_segx_path, remove_words)


def data_read(path):
    """
    @description: 读取分词数据
    @input: path-分词数据路径
    @return: word-分词list
    """
    with open(path, 'r', encoding='utf-8') as f:
        word = []
        for line in f:
            word += line.split()
    print('%s data reading is finished' % path)
    return word

def vocab_generate(voacb_path, word_item):
    """
    @description: 读取分词数据
    @input: path-分词数据路径
    @return: word-分词list
    """
    with open(voacb_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(word_item):
            key = item[0]
            f.write('%s %s' % (key, i))
            f.write('\n')

def vocab_build(voacb_path, train_segx_path, train_segy_path, test_segx_path):
    """
    @description: 读取训练集和测试集分词形成分词字典库，按词频由高到低排列
    @input: voacb_path-字典路径, train_segx_path-分词后的训练集x路径, train_segy_path-分词后的训练集y路径, test_segx_path-分词后的测试集x路径
    @return: None
    """
    word_list = []
    word_list += data_read(train_segx_path)
    word_list += data_read(train_segy_path)
    word_list += data_read(test_segx_path)
    word_item = Counter(word_list)
    word_item = sorted(word_item.items(), key=lambda x: x[1], reverse=True)
    vocab_generate(voacb_path, word_item)


if __name__ == '__main__':
    train_data_path = '../datasets/AutoMaster_TrainSet.csv'
    test_data_path = '../datasets/AutoMaster_TestSet.csv'
    train_segx_path = '../datasets/train_set.seg_x.txt'
    train_segy_path = '../datasets/train_set.seg_y.txt'
    test_segx_path = '../datasets/test_set.seg_x.txt'
    stopword_path = '../datasets/stop_words.txt'
    voacb_path = '../datasets/voacb.txt'
    REMOVE_WORDS = ['|', '[', ']', '语音', '图片']
    stop_words = read_stopword(stopword_path)
    remove_words = stop_words.union(set(REMOVE_WORDS))
    # 建立训练集和测试集分词
    wordvec_build(train_data_path, test_data_path, train_segx_path, train_segy_path, test_segx_path, remove_words)
    # 由训练集和测试集分词建立字典
    vocab_build(voacb_path, train_segx_path, train_segy_path, test_segx_path)


