import pandas as pd
import torch
from torch import nn
from utils import print_bar
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from dataloader import load_cut_data, create_dataloader


class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        self.lstm1 = nn.LSTM(config.embed_size, config.hidden_size,
                             num_layers=config.num_layers,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.5)
        self.lstm2 = nn.LSTM(8 * config.hidden_size, config.hidden_size,
                             num_layers=config.num_layers,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Linear(8 * config.hidden_size, config.hidden_size),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(config.hidden_size, 2))

    def soft_align_attention(self, x1, x2, mask1, mask2):
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        return x1_align, x2_align

    def submul(self, x1, x2):
        sub = x1 - x2
        mul = x1 * x2
        return torch.cat([sub, mul], axis=-1)

    def pool(self, x):
        # x = [batch_size, seq_len, 2 * hidden_size]
        avg_pooled = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        max_pooled = self.max_pool(x.transpose(1, 2)).squeeze(-1)
        return torch.cat([avg_pooled, max_pooled], axis=-1)

    def forward(self, x1, x2, text_len1, text_len2):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)

        packed_emb1 = pack_padded_sequence(emb1, text_len1, batch_first=True, enforce_sorted=False)
        packed_emb2 = pack_padded_sequence(emb2, text_len2, batch_first=True, enforce_sorted=False)
        o1, _ = self.lstm1(packed_emb1)
        o2, _ = self.lstm1(packed_emb2)
        o1, _ = pad_packed_sequence(o1, batch_first=True, total_length=config.max_len)
        o2, _ = pad_packed_sequence(o2, batch_first=True, total_length=config.max_len)

        mask1 = x1.eq(config.pad_idx)
        mask2 = x2.eq(config.pad_idx)

        x1_align, x2_align = self.soft_align_attention(o1, o2, mask1, mask2)
        x1_combined = torch.cat([o1, x1_align, self.submul(o1, x1_align)], axis=-1)
        x2_combined = torch.cat([o2, x2_align, self.submul(o2, x2_align)], axis=-1)

        packed_x1_combined = pack_padded_sequence(x1_combined, text_len1, batch_first=True, enforce_sorted=False)
        packed_x2_combined = pack_padded_sequence(x2_combined, text_len2, batch_first=True, enforce_sorted=False)
        x1_compose, _ = self.lstm2(packed_x1_combined)
        x2_compose, _ = self.lstm2(packed_x2_combined)
        x1_compose, _ = pad_packed_sequence(x1_compose, batch_first=True, total_length=config.max_len)
        x2_compose, _ = pad_packed_sequence(x2_compose, batch_first=True, total_length=config.max_len)

        x1_pooled = self.pool(x1_compose)
        x2_pooled = self.pool(x2_compose)
        combined_feature = torch.cat([x1_pooled, x2_pooled], axis=-1)
        sim = self.fc(combined_feature)
        return sim


def train(net, epochs, loss_fn, train_dl, valid_dl, optimizer, device, dataset_name, rnn=False):
    print_bar()
    print('Start Training...')
    total_train_loss = []
    total_valid_loss = []
    best_acc = 0
    for epoch in range(epochs):
        net.train()
        epoch_train_loss = 0
        count = 0
        for x1, x2, len1, len2, y in train_dl:
            optimizer.zero_grad()
            x1 = x1.to(device)
            x2 = x2.to(device)
            if rnn:
                y_out = net(x1, x2, len1, len2)
            else:
                y_out = net(x1, x2)
            loss = loss_fn(y_out, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == y.numpy()).mean()
        train_acc = count / len(train_dl)
        train_loss = epoch_train_loss / len(train_dl)
        valid_loss, valid_acc = evaluate(net, loss_fn, valid_dl, device, rnn)
        model_path = f'../model/ESIM/{dataset_name}_esim.model'
        if valid_acc > best_acc:
            torch.save(net.state_dict(), model_path)
            best_acc = valid_acc
            print(f'model saved at {model_path}')
        print_bar()
        print(f'Epoch: {epoch+1}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}')
        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
    return total_train_loss, total_valid_loss


def evaluate(net, loss_fn, valid_dl, device, rnn):
    net.eval()
    valid_loss = 0
    count = 0
    with torch.no_grad():
        for x1, x2, len1, len2, y in valid_dl:
            x1 = x1.to(device)
            x2 = x2.to(device)
            if rnn:
                y_out = net(x1, x2, len1, len2)
            else:
                y_out = net(x1, x2)
            loss = loss_fn(y_out, y.to(device))
            valid_loss += loss.item()
            y_pred = torch.max(y_out, 1)[1]
            count += (y_pred.cpu().numpy() == y.numpy()).mean()
    return valid_loss / len(valid_dl), count / len(valid_dl)


def predict(net, test_dl, device, rnn=False):
    y_pred = []
    for x1, x2, len1, len2 in test_dl:
        x1 = x1.to(device)
        x2 = x2.to(device)
        if rnn:
            y_out = net(x1, x2, len1, len2)
        else:
            y_out = net(x1, x2)
        y_pred.extend(torch.max(y_out.cpu(), 1)[1].tolist())
    return y_pred


if __name__ == '__main__':
    for dataset_name in config.dataset_cut_path:
        train_df, dev_df, test_df = load_cut_data(dataset_name)
        train_df = train_df[:1000]
        dev_df = dev_df[:1000]
        train_dl = create_dataloader(train_df)
        dev_dl = create_dataloader(dev_df)
        test_dl = create_dataloader(test_df, train=False)

        device = config.device
        net = ESIM().to(device)
        epochs = 10
        lr = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        train_loss, valid_loss = train(net, epochs, loss_fn, train_dl, dev_dl, optimizer, device, dataset_name, rnn=True)
        y_pred = predict(net, test_dl, device, rnn=True)
        saved_path = f'../submit/esim/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')
