import pandas as pd
import numpy as np
import pickle
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataloader import load_raw_data


def edit_distance(row):
    s = row['text1']
    t = row['text2']
    m, n = len(s), len(t)
    dist = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        dist[i][0] = i
    for j in range(1, n+1):
        dist[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                dist[i][j] = dist[i-1][j-1]
            else:
                dist[i][j] = min(dist[i-1][j-1],
                                 dist[i-1][j],
                                 dist[i][j-1]) + 1
    return dist[-1][-1]


def common_word(row):
    sset = set(row['cut_text1'].split(' '))
    tset = set(row['cut_text2'].split(' '))
    return len(sset & tset)


def common_char(row):
    sset = set(list(row['text1']))
    tset = set(list(row['text2']))
    return len(sset & tset)


def cut_text(text):
    with open(config.stopwords_path, 'r') as f:
        stopwords = []
        for word in f.readlines():
            stopwords.append(word.strip())
    remove_char_list = list('[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+')
    return ' '.join([word for word in jieba.cut(text) if word not in remove_char_list and word not in stopwords])


def feature_extraction(df):
    # 句子A包含的字符个数
    df['text1_len'] = df['text1'].apply(len)
    # 句子B包含的字符个数
    df['text2_len'] = df['text2'].apply(len)
    # 句子A与句子B的编辑距离
    df['edit_distance'] = df.apply(edit_distance, axis=1)
    # 句子A与句子B共有单词的个数
    df['cut_text1'] = df['text1'].apply(cut_text)
    df['cut_text2'] = df['text2'].apply(cut_text)
    df['common_word'] = df.apply(common_word, axis=1)
    # 句子A与句子B共有字符的个数
    df['common_char'] = df.apply(common_char, axis=1)
    # 句子A与句子B共有单词的个数 / 句子A字符个数
    df['common_1'] = df.apply(lambda row: row['common_word'] / len(row['text1']), axis=1 )
    # 句子A与句子B共有单词的个数 / 句子B字符个数
    df['common_2'] = df.apply(lambda row: row['common_word'] / len(row['text2']), axis=1)


def tfidf_extraction(vec, df, saved_path):
    # 计算TFIDF，并对句子A和句子B进行特征转换
    text1_tfidf = vec.transform(df['cut_text1'])
    text2_tfidf = vec.transform(df['cut_text2'])
    np.save(saved_path.split('.tsv')[0] + '_cut_text1.tfidf', text1_tfidf)
    np.save(saved_path.split('.tsv')[0] + '_cut_text2.tfidf', text2_tfidf)
    # 计算句子A与句子B的TFIDF向量的内积距离
    df['tfidf_sim'] = [cosine(tfidf1.toarray(), tfidf2.toarray()) for tfidf1, tfidf2 in zip(text1_tfidf, text2_tfidf)]


if __name__ == '__main__':
    for dataset_name in config.dataset_path:
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        # test_df = test_df[:100]

        feature_extraction(train_df)
        feature_extraction(dev_df)
        feature_extraction(test_df)

        vec = TfidfVectorizer()
        train_corpus = pd.concat([train_df['cut_text1'] + train_df['cut_text2']], axis=0)
        vec.fit(train_corpus)
        pickle.dump(vec, open(f'../model/{dataset_name}_tfidf.model', 'wb'))
        tfidf_extraction(vec, train_df, config.dataset_path[dataset_name]['train'])
        tfidf_extraction(vec, dev_df, config.dataset_path[dataset_name]['dev'])
        tfidf_extraction(vec, test_df, config.dataset_path[dataset_name]['test'])

        train_df.fillna(0, inplace=True)
        dev_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)

        features = ['text1_len', 'text2_len', 'edit_distance', 'common_word', 'common_char', 'common_1', 'common_2',
                    'tfidf_sim']

        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(train_df[features], train_df['label'])
        train_score = pipe.score(train_df[features], train_df['label'])
        val_score = pipe.score(dev_df[features], dev_df['label'])
        print(dataset_name + ' train score:', train_score)
        print(dataset_name + ' valid score:', val_score)

        y_pred = pipe.predict(test_df[features])
        df_pred = pd.DataFrame(y_pred, columns=['prediction'])
        submit_path = '../submit/tfidf/'
        df_pred.to_csv(submit_path + dataset_name + '.tsv', index_label='index', sep='\t')
