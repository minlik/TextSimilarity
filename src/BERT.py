import pandas as pd
from transformers import BertTokenizer, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn

import config
from dataloader import load_raw_data, create_bert_dataloader
from utils import print_bar


class Bert(nn.Module):
    def __init__(self, bert, hidden_size):
        super(Bert, self).__init__()
        self.bert = bert
        self.clf = nn.Linear(hidden_size, 2)

    def forward(self, encodings, device):
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        output = self.bert(input_ids, attention_mask).pooler_output
        return self.clf(output)


def tokenize(df, dataset_name):
    max_len = 2 * config.max_len
    if dataset_name == 'paws-x-zh':
        max_len = 4 * config.max_len
    return tokenizer(list(df['text1']), list(df['text2']), truncation='longest_first', padding='max_length', max_length=max_len)


def bert_train(bert, loss_fn, train_dl, valid_dl, epochs, optimizer, scheduler, device):
    print_bar()
    print('Start Training...')
    total_train_loss = []
    total_valid_loss = []
    best_acc = 0
    for epoch in range(epochs):
        bert.train()
        epoch_train_loss = 0
        count = 0
        for encodings in train_dl:
            optimizer.zero_grad()
            y_out = bert(encodings, device)
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
        valid_loss, valid_acc = evaluate(bert, loss_fn, valid_dl, device)
        model_path = f'../model/Bert/{dataset_name}_bert.model'
        # model_path = f'/content/drive/MyDrive/coggle/model/Bert/{dataset_name}_bert.model'
        print_bar()
        print(f'Epoch: {epoch+1}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}')
        if valid_acc > best_acc:
            torch.save(bert.state_dict(), model_path)
            best_acc = valid_acc
            print(f'model saved at {model_path}')
        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
    return total_train_loss, total_valid_loss


def evaluate(bert, loss_fn, valid_dl, device):
    bert.eval()
    valid_loss = 0
    count = 0
    with torch.no_grad():
        for encodings in valid_dl:
            labels = encodings['label'].to(device)
            y_out = bert(encodings, device)
            loss = loss_fn(y_out, labels)
            valid_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == labels.cpu().numpy()).mean()
    return valid_loss / len(valid_dl), count / len(valid_dl)


def bert_predict(bert, test_dl, device):
    bert = bert.to(device)
    bert.eval()
    y_pred = []
    for encodings in test_dl:
        y_out = bert(encodings, device)
        y_pred.extend(torch.max(y_out, 1)[1].tolist())
    return y_pred


if __name__ == '__main__':
    model_path = 'hfl/chinese-roberta-wwm-ext'
    # model_path = '/home/kuan/bert/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    bert_config = BertConfig.from_pretrained(model_path)
    bert_model = Bert(model, bert_config.hidden_size).to(config.device)
    for dataset_name in config.dataset_path:
        train_df, dev_df, test_df = load_raw_data(dataset_name)
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        train_encodings = tokenize(train_df, dataset_name)
        dev_encodings = tokenize(dev_df, dataset_name)
        test_encodings = tokenize(test_df, dataset_name)
        train_dl = create_bert_dataloader(train_encodings, list(train_df['label']))
        dev_dl = create_bert_dataloader(dev_encodings, list(dev_df['label']))
        test_dl = create_bert_dataloader(test_encodings)

        optimizer = AdamW(bert_model.parameters(), lr=3e-5)
        len_dataset = len(train_df)
        epochs = 5
        total_steps = (len_dataset // config.batch_size) * epochs
        warmup_ratio = 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio * total_steps,
                                                    num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss()
        train_loss, valid_loss = bert_train(bert_model, loss_fn, train_dl, dev_dl, epochs,  optimizer, scheduler, config.device)

        y_pred = bert_predict(bert_model, test_dl, config.device)
        saved_path = f'../submit/bert/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')