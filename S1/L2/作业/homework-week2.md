# Homework-week2
## 1. 通过gensim训练词向量
### 1.1 利用分词后的项目数据生成训练词向量用的训练数据
### 1.2 保存词向量训练数据
### 1.3 应用gensim中Word2Vec或Fasttext训练词向量
### 1.4 保存训练好的词向量

## 2. 构建embedding_matrix
读取上步计算词向量和构建的vocab词表，以vocab中的index为key值构建embedding_matrix
eg: embedding_matrix[i] = [embedding_vector]