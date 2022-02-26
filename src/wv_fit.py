import config
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

convert_cols = ['mean_pooling1', 'mean_pooling2', 'max_pooling1', 'max_pooling2','idf_pooling1', 'idf_pooling2', 'sif_pooling1', 'sif_pooling2']


def converter(instr):
    return np.fromstring(instr[1:-1], sep=' ')


def load_data(path, cols):
    return pd.read_csv(path, sep='\t', converters={col: converter for col in cols})


if __name__ == '__main__':
    for dataset_name in config.dataset_path:
        train_path = f'../data/{dataset_name}/train_features.tsv'
        dev_path = f'../data/{dataset_name}/dev_features.tsv'
        test_path = f'../data/{dataset_name}/test_features.tsv'
        train_df = load_data(train_path, convert_cols)
        dev_df = load_data(dev_path, convert_cols)
        test_df = load_data(test_path, convert_cols)

        # train_save_features = ['cut_text1', 'cut_text2', 'label']
        # test_save_features = ['cut_text1', 'cut_text2']
        # train_df[train_save_features].to_csv(config.dataset_cut_path[dataset_name]['train'], sep='\t', index=False)
        # dev_df[train_save_features].to_csv(config.dataset_cut_path[dataset_name]['dev'], sep='\t', index=False)
        # test_df[test_save_features].to_csv(config.dataset_cut_path[dataset_name]['test'], sep='\t', index=False)
        # print(f'{dataset_name} data saved!')

        for col1, col2 in [('mean_pooling1', 'mean_pooling2'), ('max_pooling1', 'max_pooling2'), ('idf_pooling1', 'idf_pooling2'), ('sif_pooling1', 'sif_pooling2')]:
            name = col1.split('_')[0]
            feature_name = f'{name}_sim'
            for df in [train_df, dev_df, test_df]:
                df[feature_name] = df.apply(lambda row: (1 - cosine(row[col1], row[col2])) / 2, axis=1)

        features = ['mean_sim', 'max_sim', 'idf_sim', 'sif_sim']
        thresholds = np.arange(0, 1, 0.1)
        for sim_feature in features:
            accuracy = []
            best_acc = 0
            best_threshold = -1
            for threshold in thresholds:
                y_pred = (train_df[sim_feature] > threshold).astype(int)
                y_true = train_df['label']
                acc = np.mean(y_pred == y_true)
                accuracy.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
            dev_y_pred = (dev_df[sim_feature] > best_threshold).astype(int)
            dev_y_true = dev_df['label']
            dev_acc = np.mean(dev_y_true == dev_y_pred)
            print(f'{sim_feature} best threshold: {best_threshold:.4f}')
            print(f'{dataset_name} train score: {best_acc:.4f}')
            print(f'{dataset_name} valid score: {dev_acc:.4f}')
            # plt.plot(thresholds, accuracy)
            # plt.title(f'{sim_feature}: accuracy vs threshold')
            # plt.xlabel('threshold')
            # plt.ylabel('accuracy')
            # plt.show()

        model = LogisticRegression()
        model.fit(train_df[features], train_df['label'])
        train_score = model.score(train_df[features], train_df['label'])
        dev_score = model.score(dev_df[features], dev_df['label'])
        print(f'{dataset_name} multi_sim train score:', train_score)
        print(f'{dataset_name} multi_sim valid score:', dev_score)

        y_pred = model.predict(test_df[features])
        df_pred = pd.DataFrame(y_pred, columns=['prediction'])
        submit_path = f'../submit/wv_sim/{dataset_name}.tsv'
        df_pred.to_csv(submit_path, sep='\t', index_label='index')
