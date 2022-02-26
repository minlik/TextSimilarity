本项目是对 [千言数据集：文本相似度](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) 比赛研究过程的记录。

项目中用到的相关模型的搭建、训练及预测代码位于 `./src` 文件夹内，具体文件说明如下。

1. `tfidf-feature.py`: 文本 TFIDF 及统计特征的提取，以及相似度计算；
2. `word2vec.py`: 使用 gensim 训练中文词向量；
3. `wv_encoding.py`: 利用词向量完成 mean / max / sif 句子向量的计算，完成相似度预测；
4. `SiamCNN.py`: 搭建 Siamese TextCNN 模型，完成相似度预测；
5. `SiamGRU`: 搭建 Siamese GRU 模型，完成相似度预测；
6. `SEIM.py`: 搭建 ESIM 模型，完成相似度预测；
7. `BERT.py`: 对 BERT 模型的 NSP 任务进行 finetune，并完成相似度预测；
8. `SentenceBert.py`: 利用 Sentence Bert 完成句子向量的编码，并完成相似度预测。

以上内容还在完善中。