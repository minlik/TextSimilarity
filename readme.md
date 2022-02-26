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
8. `BertCNN.py`: Bert + BiLSTM + TextCNN 完成句子向量的编码，并完成相似度预测。



Result：

| Model         | Score  | bq_corpus | lcqmc  | paws-x |
| ------------- | ------ | --------- | ------ | ------ |
| Siam Gru      | 0.7136 | 0.7379    | 0.7944 | 0.6085 |
| Siam CNN      | 0.6884 | 0.7198    | 0.7998 | 0.5455 |
| ESIM          | 0.6909 | 0.7542    | 0.7281 | 0.5905 |
| Bert          | 0.8037 | 0.841     | 0.8575 | 0.7125 |
| Sentence Bert | 0.7984 | 0.8352    | 0.8406 | 0.7195 |
| BertCNN       | 0.8361 | 0.8449    | 0.8645 | 0.799  |