import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

import config
from dataloader import load_raw_data
from tfidf_feature import cut_text


# 将N*100的矩阵进行max-pooling / mean-pooling编码，转为100维度。
def pooling(model, sentence, pooling_type='mean', idf=None, vocab=None):
    embeddings = []
    idf_weights = []
    for word in sentence.split():
        try:
            embeddings.append(model.wv[word])
        except KeyError:
            continue
        if pooling_type == 'idf':
            idf_weights.append(idf[vocab[word]] if word in vocab else 0.0)
    if len(embeddings) == 0:
        return np.zeros((config.embed_size, ))
    if pooling_type == 'max':
        return np.max(embeddings, axis=0)
    elif pooling_type == 'mean':
        return np.mean(embeddings, axis=0)
    elif pooling_type == 'idf':
        return np.dot(np.transpose(embeddings), idf_weights)
    else:
        raise 'please input correct pooling method(mean/max).'


def get_sif_weights(df, a=0.05):
    counter = defaultdict(int)
    weights = {}
    total = 0
    for sentence in df:
        for word in sentence.split():
            total += 1
            counter[word] += 1
    for word in counter:
        p_word = counter[word] / total
        weights[word] = a / (a + p_word)
    return weights


def sif_pooling(model, sentence, weights):
    embeddings = np.zeros(config.embed_size)
    total = 0
    for word in sentence.split():
        try:
            embeddings += model.wv[word] * weights[word]
            total += 1
        except KeyError:
            continue
    return embeddings / total


def remove_pc(embs):
    embs = np.array(embs)
    svd = TruncatedSVD(n_components=1, n_iter=7)
    svd.fit(embs)
    pc = svd.components_
    return embs - embs.dot(pc.transpose()) * pc


if __name__ == '__main__':
    wv_model = Word2Vec.load('../model/w2v.model')
    for dataset_name in config.dataset_path:
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        # test_df = test_df[:100]
        tfidf_path = f'../model/{dataset_name}_tfidf.model'
        vec = pickle.load(open(tfidf_path, 'rb'))
        idf = vec.idf_
        vocab = vec.vocabulary_
        for df, name in [(train_df, 'train'), (dev_df, 'dev'), (test_df, 'test')]:
            df['cut_text1'] = df['text1'].apply(cut_text)
            df['cut_text2'] = df['text2'].apply(cut_text)
            for pooling_type in ['mean', 'max']:
                df[f'{pooling_type}_pooling1'] = df['cut_text1'].apply(lambda sentence: pooling(wv_model, sentence, pooling_type))
                df[f'{pooling_type}_pooling2'] = df['cut_text2'].apply(lambda sentence: pooling(wv_model, sentence, pooling_type))
            df['idf_pooling1'] = df['cut_text1'].apply(lambda sentence: pooling(wv_model, sentence, pooling_type, idf, vocab))
            df['idf_pooling2'] = df['cut_text2'].apply(lambda sentence: pooling(wv_model, sentence, pooling_type, idf, vocab))

            sif_weights1 = get_sif_weights(df['cut_text1'])
            df['sif_pooling1'] = df['cut_text1'].apply(lambda sentence: sif_pooling(wv_model, sentence, sif_weights1))
            # sif_emb1 = remove_pc(df['sif_pooling1'].tolist())
            # df['sif_pooling1'] = pd.Series(list(sif_emb1))

            sif_weights2 = get_sif_weights(df['cut_text2'])
            df['sif_pooling2'] = df['cut_text2'].apply(lambda sentence: sif_pooling(wv_model, sentence, sif_weights2))
            # sif_emb2 = remove_pc(df['sif_pooling2'].tolist())
            # df['sif_pooling2'] = pd.Series(list(sif_emb2))

            saved_path = f'../data/{dataset_name}/{name}_features.tsv'
            print(f'saving {saved_path}...')
            df.to_csv(saved_path, sep='\t', index=False)

