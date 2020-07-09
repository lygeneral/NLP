# L1(step1): 根据数据集进行分词
import os
import pandas as pd

from utils.tokenizer import segment


def read_stopword(path):
    """
    @description: 从文件中读取停用词并返回set集合
    @param: path-停用词路径
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
    @param: word_list-原始list
    @return: word_list-去除无效词、字符后的list
    """
    word_list = [word for word in word_list if word not in remove_words]
    return word_list

def parse_data(train_path, test_path):
    """
    @description: 读取训练数据和测试数据
    @param: train_path-训练数据路径，test_path-测试数据路径
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


def save_data(data, path, remove_words, label = False):
    """
    @description: 输入数据并在去除停用词及无效词后进行分词
    @param: data-数据集, path-写入文件的路径, remove_words-待去除的无效词
    @return: None
    """
    with open(path, 'w', encoding='utf-8') as f:
        count = 0
        for line in data:
            if isinstance(line, str):
                seg_list = segment(line.strip())
                seg_list = remove_word(seg_list, remove_words)
                ## 预处理数据为最后一列标签值需填充数据
                if label:
                    if len(seg_list) > 0:
                        seg_line = ' '.join(seg_list)
                        f.write('%s' % seg_line)
                        f.write('\n')
                    else:
                        f.write('随时 联系')
                        f.write('\n')
                    count += 1
                else:
                    if len(seg_list) > 0:
                        seg_line = ' '.join(seg_list)
                        f.write('%s' % seg_line)
                        f.write('\n')
                        count += 1
    print('%s is finished, length is ' % path, count)


def wordvec_build(train_data_path, test_data_path, train_segx_path, train_segy_path, test_segx_path, remove_words):
    """
    @description: 将训练集和测试集的数据进行分词
    @param: train_data_path-数据集路径, test_data_path-测试集路径, remove_words-待去除的无效词
    @param: train_segx_path-分词后的训练集x路径, train_segy_path-分词后的训练集y路径, test_segx_path-分词后的测试集x路径
    @return: None
    """
    train_x, train_y, test_x, _ = parse_data(train_data_path, test_data_path)
    save_data(train_x, train_segx_path, remove_words)
    save_data(train_y, train_segy_path, remove_words, label=True)
    save_data(test_x, test_segx_path, remove_words)



if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REMOVE_WORDS = ['|', '[', ']', '语音', '图片']
    train_data_path = '{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR)
    test_data_path = '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR)
    train_segx_path = '{}/datasets/train_set.seg_x.txt'.format(BASE_DIR)
    train_segy_path = '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR)
    test_segx_path = '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR)
    stopword_path = '{}/datasets/stop_words.txt'.format(BASE_DIR)

    stop_words = read_stopword(stopword_path)
    remove_word_list = stop_words.union(set(REMOVE_WORDS))
    # 建立训练集和测试集分词
    wordvec_build(train_data_path, test_data_path, train_segx_path, train_segy_path, test_segx_path, remove_word_list)