
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import time

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('%s函数运行时间：%.8f' % (func.__name__, end_time - start_time))
        return res
    return wrapper

def read_data(path):
    """
    @description: 从文本中读取数据
    @param: path-文本路径
    @return: lines-字符串list
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append((line))
    return lines
    print('%s is read' % path)

def save_data(data, path):
    """
    @description: 将数据写入文本中
    @param: path-文本路径
    @return: None
    """
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write('%s\n' % line.strip())
    print('%s is saved' % path)

@time_decorator
def word2vec_build(sentence_path, w2v_bin_path, min_count=100):
    """
    @description: word2vec训练词向量
    @param: sentence_path-数据的路径, w2v_bin_path-保存词向量（模型）的路径
    @return: None
    @Word2Vec:
    sg： 用于设置训练算法，默认为0，对应CBOW（上下文预测中心词）算法；sg=1则采用skip-gram算法。
    sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。
    size：是指特征向量的维度，默认为100。
    window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少。
    min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    iter： 迭代次数，默认为5。
    """
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)

@time_decorator
def fasttext_build(sentence_path, ft_bin_path, min_count=100):
    """
    @description: fasttext训练词向量
    @param: sentence_path-数据的路径, ft_bin_path-保存词向量（模型）的路径
    @return: None
    """
    ft = FastText(sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=5)
    ft.wv.save_word2vec_format(ft_bin_path, binary=True)
    print("save %s ok." % ft_bin_path)

def model_test(model, kw1, kw2):
    """
    @description: 计算两词的相似度及与其相似的词
    @param: model-模型, kw1-词1, kw2-词2
    @return: None
    """
    print('%s and %s similarity: %s' % (kw1, kw2, model.wv.similarity(kw1, kw2)))
    print('%s similar voacb: %s' % (kw1, model.wv.similar_by_word(kw1)))
    print('%s similar voacb: %s' % (kw2, model.wv.similar_by_word(kw2)))

def embedding_matrix(model, vocab_path):
    """
    @description: 以vocab中的index为key值构建embeddingmatrix
    @param: model-模型, vocab_path-字典路径
    @return: embedding_matrix-index及对应词向量
    """
    embedding_matrix = {}
    for word in model.vocab:
        embedding_matrix[word] = model[word]
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if line[0] in embedding_matrix.keys():
                embedding_matrix[line[1]] = embedding_matrix.pop(line[0])
    return embedding_matrix

if __name__ == '__main__':
    """
    @description: 训练词向量，保存、加载模型，并计算词的相似度
    word2vec保存和加载模型方法（save方式保存的模型可继续训练，save_word2vec_format速度快内存小）：
    1.model.save('w2v.model')->model = Word2Vec.load('w2v.model')
    2.model.wv.save_word2vec_format('w2v.bin')
    ->KeyedVectors.load_word2vec_format('w2v.bin')
    """
    train_segx_path = '../datasets/train_set.seg_x.txt'
    train_segy_path = '../datasets/train_set.seg_y.txt'
    test_segx_path = '../datasets/test_set.seg_x.txt'
    sentences_path = '../datasets/sentences.txt'
    w2v_bin_path = 'w2v.bin'
    ft_bin_path = 'ft.bin'
    voacb_path = '../datasets/voacb.txt'
    lines = read_data(train_segx_path)
    lines += read_data(train_segy_path)
    lines += read_data(test_segx_path)
    save_data(lines, sentences_path)
    word2vec_build(sentences_path, w2v_bin_path)
    fasttext_build(sentences_path, ft_bin_path)

    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path,binary=True)
    model_test(w2v_model,'汽车','车')

    ft_model = KeyedVectors.load_word2vec_format(ft_bin_path,binary=True)
    model_test(ft_model,'汽车','车')

    mt = embedding_matrix(w2v_model, voacb_path)
