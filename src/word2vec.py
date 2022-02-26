import pandas as pd
from gensim.models import Word2Vec
from tfidf_feature import cut_text
import config
from dataloader import load_raw_data


class MySentences:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for sentence in self.data:
            yield str(sentence).split()


if __name__ == '__main__':
    train_data = None
    for dataset_name in config.dataset_path:
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        for df in [train_df, dev_df, test_df]:
            df['cut_text1'] = df['text1'].apply(cut_text)
            df['cut_text2'] = df['text2'].apply(cut_text)
        data = pd.concat([train_df['cut_text1'],
                          train_df['cut_text2'],
                          dev_df['cut_text1'],
                          dev_df['cut_text2'],
                          test_df['cut_text1'],
                          test_df['cut_text2']],
                         axis=0)
        if train_data is None:
            train_data = data
        else:
            train_data = pd.concat([train_data, data], axis=0)

    sentences = MySentences(train_data)
    model = Word2Vec(sentences, vector_size=config.embed_size, workers=-1, epochs=5)

    model.save('../model/w2v.model')
