import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


def load_raw_data(dataset_name):
    if dataset_name == 'bq_corpus':
        train_df = pd.read_csv(config.dataset_path[dataset_name]['train'], sep='\t\n', header=None)
        text1 = np.asarray([i.split('\t')[0] for i in train_df[0]])
        text2 = np.asarray([i.split('\t')[1] for i in train_df[0]])
        label = np.asarray([i.split('\t')[2] for i in train_df[0]])
        train_df['text1'] = text1
        train_df['text2'] = text2
        train_df['label'] = label.astype(int)
        train_df.drop(train_df.columns[0], axis=1, inplace=True)
    else:
        train_df = pd.read_table(config.dataset_path[dataset_name]['train'], header=None)
        train_df.columns = config.train_data_style

    dev_df = pd.read_table(config.dataset_path[dataset_name]['dev'], header=None)
    dev_df.columns = config.dev_data_style

    test_df = pd.read_table(config.dataset_path[dataset_name]['test'], header=None)
    test_df.columns = config.test_data_style

    train_df['text1'] = train_df['text1'].astype(str)
    train_df['text2'] = train_df['text2'].astype(str)
    dev_df['text1'] = dev_df['text1'].astype(str)
    dev_df['text2'] = dev_df['text2'].astype(str)
    test_df['text1'] = test_df['text1'].astype(str)
    test_df['text2'] = test_df['text2'].astype(str)
    return train_df, dev_df, test_df


def load_cut_data(dataset_name):
    train_df = pd.read_csv(config.dataset_cut_path[dataset_name]['train'], sep='\t')
    dev_df = pd.read_csv(config.dataset_cut_path[dataset_name]['dev'], sep='\t')
    test_df = pd.read_csv(config.dataset_cut_path[dataset_name]['test'], sep='\t')
    return train_df, dev_df, test_df


class MyDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text1 = str(self.df.iloc[idx]['cut_text1']).split()
        text2 = str(self.df.iloc[idx]['cut_text2']).split()
        out1 = [config.wv_model.wv.key_to_index.get(t1, config.unk_idx) for t1 in text1]
        out2 = [config.wv_model.wv.key_to_index.get(t2, config.unk_idx) for t2 in text2]
        len1 = min(len(text1), config.max_len)
        len2 = min(len(text2), config.max_len)
        if len(out1) > config.max_len:
            out1 = out1[:config.max_len]
        else:
            out1 += [config.pad_idx] * (config.max_len - len(out1))
        if len(out2) > config.max_len:
            out2 = out2[:config.max_len]
        else:
            out2 += [config.pad_idx] * (config.max_len - len(out2))
        if self.train:
            return torch.tensor(out1), torch.tensor(out2), len1, len2, torch.tensor(self.df.iloc[idx]['label'])
        return torch.tensor(out1), torch.tensor(out2), len1, len2


def create_dataloader(df, train=True):
    ds = MyDataset(df, train=train)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=train)
    return dl


class BertDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['label'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def create_bert_dataloader(encodings, labels=None):
    ds = BertDataset(encodings, labels)
    shuffle = True if labels is not None else False
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)
    return dl


class SentenceBertDataset(Dataset):
    def __init__(self, encodings1, encodings2, labels=None):
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        item1 = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        if self.labels is not None:
            return item1, item2, torch.tensor(self.labels[idx])
        return item1, item2

    def __len__(self):
        return len(self.encodings1['input_ids'])


def create_sbert_dataloader(encodings1, encodings2, labels=None):
    ds = SentenceBertDataset(encodings1, encodings2, labels)
    shuffle = True if labels is not None else False
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)
    return dl
