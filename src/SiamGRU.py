import pandas as pd
import torch
from torch import nn
from utils import print_bar

import config
from dataloader import load_cut_data, create_dataloader


class SiamGRU(nn.Module):
    def __init__(self):
        super(SiamGRU, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        self.gru = nn.GRU(config.embed_size,
                          config.hidden_size,
                          config.num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.5)
        self.clf = nn.Linear(8 * config.hidden_size, 16)

    def forward_once(self, x, text_len):
        embed = self.embedding(x)
        output, _ = self.gru(embed)
        avg_out = torch.mean(output, 1)
        return avg_out

    def cross_layer(self, x1, x2):
        f1 = torch.abs(x1 - x2)
        f2 = torch.mul(x1, x2)
        return torch.cat([x1, x2, f1, f2], axis=1)

    def forward(self, x1, x2, text_len1, text_len2):
        out1 = self.forward_once(x1, text_len1)
        out2 = self.forward_once(x2, text_len2)
        cross_features = self.cross_layer(out1, out2)
        out = self.clf(cross_features)
        return out


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
        model_path = f'../model/SiamGRU/{dataset_name}_siamgru.model'
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
        # train_df = train_df[:100]
        # dev_df = dev_df[:100]
        train_dl = create_dataloader(train_df)
        dev_dl = create_dataloader(dev_df)
        test_dl = create_dataloader(test_df, train=False)

        device = config.device
        net = SiamGRU().to(device)
        epochs = 10
        lr = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        train_loss, valid_loss = train(net, epochs, loss_fn, train_dl, dev_dl, optimizer, device, dataset_name, rnn=True)
        y_pred = predict(net, test_dl, device, rnn=True)
        saved_path = f'../submit/siamgru/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')
