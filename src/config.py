import torch
import gensim

dataset_path = {
    'lcqmc':
            {'train': '../data/lcqmc/train.tsv',
             'dev': '../data/lcqmc/dev.tsv',
             'test': '../data/lcqmc/test.tsv'},
    'paws-x-zh':
            {'train': '../data/paws-x-zh/train.tsv',
             'dev': '../data/paws-x-zh/dev.tsv',
             'test': '../data/paws-x-zh/test.tsv'},
    'bq_corpus':
            {'train': '../data/bq_corpus/train.tsv',
             'dev': '../data/bq_corpus/dev.tsv',
             'test': '../data/bq_corpus/test.tsv'}
}

dataset_cut_path = {
    'lcqmc':
            {'train': '../data/lcqmc/train_cut_text.tsv',
             'dev': '../data/lcqmc/dev_cut_text.tsv',
             'test': '../data/lcqmc/test_cut_text.tsv'},
    'paws-x-zh':
            {'train': '../data/paws-x-zh/train_cut_text.tsv',
             'dev': '../data/paws-x-zh/dev_cut_text.tsv',
             'test': '../data/paws-x-zh/test_cut_text.tsv'},
    'bq_corpus':
            {'train': '../data/bq_corpus/train_cut_text.tsv',
             'dev': '../data/bq_corpus/dev_cut_text.tsv',
             'test': '../data/bq_corpus/test_cut_text.tsv'}
}

train_data_style = ['text1', 'text2', 'label']
dev_data_style = ['text1', 'text2', 'label']
test_data_style = ['text1', 'text2']

stopwords_path = '../data/cn_stopwords.txt'

embed_size = 300

# params for SiamCNN / SiamLSTM model
feature_size = 100
window_sizes = [2, 3, 4, 5]
num_layers = 2
hidden_size = 256
max_len = 64

# params for training
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# params for word2vec model
wv_model = gensim.models.Word2Vec.load('../model/w2v.model')
unk_idx = len(wv_model.wv.key_to_index)
pad_idx = unk_idx + 1
wv_model.wv.key_to_index['<UNK>'] = unk_idx
wv_model.wv.key_to_index['<PAD>'] = pad_idx
vocab_size = len(wv_model.wv.key_to_index)
