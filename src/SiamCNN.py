import pandas as pd
import torch
from torch import nn
from utils import print_bar

import config
from dataloader import load_cut_data, create_dataloader


class SiamCNN(nn.Module):
    def __init__(self):
        super(SiamCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, config.feature_size, (kernel_size, config.embed_size)),
                          nn.LeakyReLU(inplace=True),
                          nn.AdaptiveAvgPool2d(1))
            for kernel_size in config.window_sizes
        ])
        self.fc = nn.Linear(config.feature_size * len(config.window_sizes), 16)
        self.clf = nn.Linear(4 * 16, 2)

    def forward_once(self, x):
        embed = self.embedding(x)
        embed.unsqueeze_(1)
        conv_out = [conv(embed) for conv in self.convs]
        out = torch.cat(conv_out, dim=1)
        out = out.view(x.size(0), -1)
        return self.fc(out)

    def cross_layer(self, x1, x2):
        f1 = torch.abs(x1 - x2)
        f2 = torch.mul(x1, x2)
        return torch.cat([x1, x2, f1, f2], axis=1)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
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
        model_path = f'../model/SiamCNN/{dataset_name}_siamcnn.model'
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
        net = SiamCNN().to(device)
        epochs = 10
        lr = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        train_loss, valid_loss = train(net, epochs, loss_fn, train_dl, dev_dl, optimizer, device, dataset_name, rnn=False)
        y_pred = predict(net, test_dl, device)
        saved_path = f'../submit/siamcnn/{dataset_name}.tsv'
        pd.DataFrame(y_pred, columns=['prediction']).to_csv(saved_path, sep='\t', index_label='index')
        print(f'{dataset_name} training and prediction finished!')
        print(f'prediction result saved path: {saved_path}')
