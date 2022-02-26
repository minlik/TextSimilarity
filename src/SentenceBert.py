import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd

import config
from utils import print_bar
from dataloader import load_raw_data, create_sbert_dataloader


class Model(nn.Module):
    def __init__(self, model_path, encoder_type='first-last-avg'):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.encoder_type = encoder_type
        self.fc = nn.Linear(self.bert.config.hidden_size * 3, 2)

    def get_output_layer(self, output):
        if self.encoder_type == 'first-last-avg':
            first = output.hiddens_states[1]
            last = output.hidden_states[-1]
            avg_first = torch.mean(first, axis=1)
            avg_last = torch.mean(last, axis=1)
            return (avg_first + avg_last) / 2

        if self.encoder_type == 'last_avg':
            last = output.hiddens_states[-1]
            return torch.mean(last, axis=1)

        if self.encoder_type == 'cls':
            return output.pooler_output

        # default encoder output
        return output.pooler_output

    def cross_layer(self, out1, out2):
        return torch.cat([out1, out2, torch.abs(out1 - out2)], axis=1)

    def forward(self, encodings1, encodings2, device):
        input_ids1 = encodings1['input_ids'].to(device)
        input_ids2 = encodings2['input_ids'].to(device)
        attention_mask1 = encodings1['attention_mask'].to(device)
        attention_mask2 = encodings2['attention_mask'].to(device)
        bert_output1 = self.bert(input_ids1, attention_mask=attention_mask1, output_hidden_states=True)
        bert_output2 = self.bert(input_ids2, attention_mask=attention_mask2, output_hidden_states=True)
        pooler1 = self.get_output_layer(bert_output1)
        pooler2 = self.get_output_layer(bert_output2)
        cross_features = self.cross_layer(pooler1, pooler2)
        out = self.fc(cross_features)
        return out


def tokenize(df):
    encodings1 = tokenizer(list(df['text1']), truncation=True, padding='max_length', max_length=config.max_len)
    encodings2 = tokenizer(list(df['text2']), truncation=True, padding='max_length', max_length=config.max_len)
    return encodings1, encodings2


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
        for encodings1, encodings2, labels in train_dl:
            optimizer.zero_grad()
            y_out = model(encodings1, encodings2, device)
            labels = labels.to(device)
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
        model_path = f'../model/SBert/{dataset_name}_sbert.model'
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
        for encodings1, encodings2, labels in valid_dl:
            y_out = model(encodings1, encodings2, device)
            labels = labels.to(device)
            loss = loss_fn(y_out, labels)
            valid_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == labels.cpu().numpy()).mean()
    return valid_loss / len(valid_dl), count / len(valid_dl)


def bert_predict(model, test_dl, device):
    model.eval()
    y_pred = []
    for encodings1, encodings2 in test_dl:
        y_out = model(encodings1, encodings2, device)
        y_pred.extend(torch.max(y_out, 1)[1].tolist())
    return y_pred


if __name__ == '__main__':
    model_path = 'hfl/chinese-roberta-wwm-ext'
    # model_path = '/home/kuan/bert/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    for dataset_name in config.dataset_path:
        model = Model(model_path, config.device).to(config.device)
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        # test_df = test_df[:100]
        train_encodings1, train_encodings2 = tokenize(train_df)
        dev_encodings1, dev_encodings2 = tokenize(dev_df)
        test_encodings1, test_encodings2 = tokenize(test_df)
        train_dl = create_sbert_dataloader(train_encodings1, train_encodings2, list(train_df['label']))
        dev_dl = create_sbert_dataloader(dev_encodings1, dev_encodings2, list(dev_df['label']))
        test_dl = create_sbert_dataloader(test_encodings1, test_encodings2)

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
        saved_path = f'../submit/sbert/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')
