import numpy as np
import pandas as pd
import torch
from torch import nn
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt


import config
from utils import print_bar
from dataloader import load_raw_data, create_bert_dataloader


transformers.logging.set_verbosity_error()


class Model(nn.Module):
    def __init__(self, model_path):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.gru = nn.LSTM(self.bert.config.hidden_size, config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.5)

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, config.feature_size, (kernel_size, config.embed_size)),
                          nn.LeakyReLU(inplace=True),
                          nn.AdaptiveMaxPool2d(1))
            for kernel_size in config.window_sizes
        ])

        self.fc = nn.Linear(len(config.window_sizes) * config.feature_size, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encodings, device):
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        text_len = torch.sum(attention_mask == 1, axis=-1).cpu()
        packed_x = pack_padded_sequence(bert_output, text_len, batch_first=True, enforce_sorted=False)
        x_gru, _ = self.gru(packed_x)
        x_emb, _ = pad_packed_sequence(x_gru, batch_first=True)

        x_emb.unsqueeze_(1)
        cnn_out = torch.cat([conv(x_emb) for conv in self.convs], axis=1)
        cnn_out = cnn_out.view(cnn_out.shape[0], -1)
        return self.fc(self.dropout(cnn_out))


def tokenize(df, dataset_name):
    max_len = 2 * config.max_len
    if dataset_name == 'paws-x-zh':
        max_len = 4 * config.max_len
    return tokenizer(list(df['text1']), list(df['text2']), truncation='longest_first', padding='max_length', max_length=max_len)


def bert_train(model, loss_fn, train_dl, valid_dl, epochs, optimizer, scheduler, device):
    print_bar()
    print('Start Training...')
    total_train_loss = []
    total_valid_loss = []
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        count = 0
        for encodings in train_dl:
            optimizer.zero_grad()
            y_out = model(encodings, device)
            labels = encodings['label'].to(device)
            loss = loss_fn(y_out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == labels.cpu().numpy()).mean()
        train_acc = count / len(train_dl)
        train_loss = epoch_train_loss / len(train_dl)
        valid_loss, valid_acc = evaluate(model, loss_fn, valid_dl, device)
        model_path = f'../model/BertCNN/{dataset_name}_bertcnn.model'
        print_bar()
        print(f'Epoch: {epoch+1}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}')
        if valid_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = valid_acc
            print(f'model saved at {model_path}')
        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
    return total_train_loss, total_valid_loss


def evaluate(model, loss_fn, valid_dl, device):
    model.eval()
    valid_loss = 0
    count = 0
    with torch.no_grad():
        for encodings in valid_dl:
            y_out = model(encodings, device)
            labels = encodings['label'].to(device)
            loss = loss_fn(y_out, labels)
            valid_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == labels.cpu().numpy()).mean()
    return valid_loss / len(valid_dl), count / len(valid_dl)


def bert_predict(model, test_dl, device):
    model.eval()
    y_pred = []
    for encodings1 in test_dl:
        y_out = model(encodings1, device)
        y_pred.extend(torch.max(y_out, 1)[1].tolist())
    return y_pred


if __name__ == '__main__':
    model_path = 'hfl/chinese-roberta-wwm-ext'
    # model_path = '/home/kuan/bert/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    for dataset_name in config.dataset_path:
        if dataset_name == 'lcqmc':
            model = Model(model_path).to(config.device)
            saved_path = f'../model/BertCNN/{dataset_name}_bertcnn.model'
            model.load_state_dict(torch.load(saved_path))
            _, _, test_df = load_raw_data(dataset_name)
            test_encodings = tokenize(test_df, dataset_name)
            test_dl = create_bert_dataloader(test_encodings)
            y_pred = bert_predict(model, test_dl, config.device)
            saved_path = f'../submit/bertcnn/{dataset_name}.tsv'
            pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
            print(f'{dataset_name} training and prediction finished!')
            print(f'prediction result saved path: {saved_path}')
            continue
        model = Model(model_path).to(config.device)
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        # test_df = test_df[:100]
        train_encodings = tokenize(train_df, dataset_name)
        dev_encodings = tokenize(dev_df, dataset_name)
        test_encodings = tokenize(test_df, dataset_name)
        train_dl = create_bert_dataloader(train_encodings, list(train_df['label']))
        dev_dl = create_bert_dataloader(dev_encodings, list(dev_df['label']))
        test_dl = create_bert_dataloader(test_encodings)

        optimizer = AdamW(model.parameters(), lr=3e-5)
        len_dataset = len(train_df)
        epochs = 5
        total_steps = (len_dataset // config.batch_size) * epochs
        warmup_ratio = 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio * total_steps,
                                                    num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss()
        train_loss, valid_loss = bert_train(model, loss_fn, train_dl, dev_dl, epochs,  optimizer, scheduler, config.device)

        y_pred = bert_predict(model, test_dl, config.device)
        saved_path = f'../submit/bertcnn/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')
