> 📋A template README.md for code accompanying a Machine Learning paper

# 后厂理工NLP

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## 项目1-汽车大师问答摘要与推理

> 本次比赛主题为[百度AI-Studio常规赛：汽车大师问答摘要与推理](https://aistudio.baidu.com/aistudio/competition/detail/3)。要求使用汽车大师提供的11万条 技师与用户的多轮对话与诊断建议报告 数据建立模型，模型需基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，综合考验模型的归纳总结与推断能力。

> 汽车大师是一款通过在线咨询问答为车主解决用车问题的APP，致力于做车主身边靠谱的用车顾问，车主用语音、文字或图片发布汽车问题，系统为其匹配专业技师提供及时有效的咨询服务。由于平台用户基数众多，问题重合度较高，大部分问题在平台上曾得到过解答。重复回答和持续时间长的多轮问询不仅会花去汽修技师大量时间，也使用户获取解决方案的时间变长，对双方来说都存在资源浪费的情况。

> 为了节省更多人工时间，提高用户获取回答和解决方案的效率，汽车大师希望通过利用机器学习对平台积累的大量历史问答数据进行模型训练，基于汽车大师平台提供的历史多轮问答文本，输出完整的建议报告和回答，让用户在线通过人工智能语义识别即时获得全套解决方案。

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Summary

### L1

> 初识词向量Word2Vec，并介绍了经典Skip-Gram和CBOW模型。Skip-Gram通过中心词预测上下文，即后验概率。CBOW通过上下文预测中心词，即先验概率。

> 首先对训练和测试数据集进行预处理，去除特殊字符。然后基于jieba分词并建立字典。  