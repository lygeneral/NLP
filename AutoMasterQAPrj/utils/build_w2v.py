# L2(step3): 把数据集所有的分词放在一起训练词向量并保存模型
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from utils.data_utils import dump_pkl
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('%s函数运行时间：%.8f' % (func.__name__, end_time - start_time))
        return res
    return wrapper

def read_lines(path, col_sep=None):
    """
    @description: 从文本中读取数据
    @param: path-文本路径
    @return: lines-字符串list
    """
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    print('%s read finished' % path)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    """
    @description: 将数据写入文本中
    @param: path-文本路径
    @return: None
    """
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)

def model_test(model, kw1, kw2):
    """
    @description: 计算两词的相似度及与其相似的词
    @param: model-模型, kw1-词1, kw2-词2
    @return: None
    """
    print('%s and %s similarity: %s' % (kw1, kw2, model.similarity(kw1, kw2)))
    print('%s similar voacb: %s' % (kw1, model.similar_by_word(kw1)))
    print('%s similar voacb: %s' % (kw2, model.similar_by_word(kw2)))

@time_decorator
def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    # test
    model_test(model, '技师', '车主')
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    """
    @description: 训练词向量，保存、加载模型，并计算词的相似度
    word2vec保存和加载模型方法（save方式保存的模型可继续训练，save_word2vec_format速度快内存小）：
    1.model.save('w2v.model')->model = Word2Vec.load('w2v.model')
    2.model.wv.save_word2vec_format('w2v.bin')
    ->KeyedVectors.load_word2vec_format('w2v.bin')
    """
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))

