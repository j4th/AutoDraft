"""
Module for working with GluonTS and models from it
"""
from pathlib import Path
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
import pandas as pd

def clean_duplicates(data, streamlit=False):
    unique = data.loc[:, 'name'].unique()
    num_unique = len(unique)
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        player_df = player_df.drop_duplicates(subset='date')
        clean = pd.concat([clean, player_df])
    return clean, num_unique

def clean_rookies_retirees(data, split_from='2018-10-03'):
    unique = data.loc[:, 'name'].unique()
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        train_df = player_df.loc[player_df.loc[:, 'date'] < split_from]
        test_df = player_df.loc[player_df.loc[:, 'date'] >= split_from]
        if train_df.shape[0] == 0:
            continue
        elif test_df.shape[0] == 0:
            continue
        clean = pd.concat([clean, player_df])
    return clean

def clean_df(data, split_from='2018-10-03', streamlit=False):
    clean = clean_rookies_retirees(data, split_from=split_from)
    clean, num_unique = clean_duplicates(clean, streamlit=streamlit)
    return clean, num_unique

def subset_df(data, column_list=None):
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    data = data.loc[:, column_list]
    return data

def split_train_test(data, split_from='2018-10-03'):
    train = data.loc[data.loc[:, 'date'] < split_from]
    test = data.loc[data.loc[:, 'date'] >= split_from]
    return train, test

def prep_df(data, split_from='2018-10-03', column_list=None, streamlit=False):
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    data, num_unique = clean_df(data, split_from=split_from, streamlit=streamlit)
    targets = assemble_target(data)
    train, test = split_train_test(data)
    train = train.loc[:, column_list]
    test = test.loc[:, column_list]
    return train, test, num_unique, targets

def generate_metadata(train_data, test_data, num_unique, index=None):
    # prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
    #                                     for name in test_data.loc[:, 'name'].unique()]
    if index is None:
        if 'date' in train_data.columns:
            index = 'date'
        elif 'gameNumber' in train_data.columns:
            index = 'gameNumber'
    if index == 'date':
        data_meta = {'num_series': num_unique,
                    'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                .shape[0] for name in train_data.loc[:, 'name'].unique()],
                    'prediction_length': 82,
                    'freq': '1D',
                    'start': [pd.Timestamp(train_data.loc[train_data.loc[:, 'name'] == name] \
                                            .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
                                            .index[0], 'date'], freq='1D')
                            for name in train_data.loc[:, 'name'].unique()]
                    }
    elif index == 'gameNumber':
        data_meta = {'num_series': num_unique,
                    'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                  .shape[0] for name in train_data.loc[:, 'name'].unique()],
                    'prediction_length': 82,
                    'freq': '1D',
                    'start': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .index[0], 'gameNumber']
                              for name in train_data.loc[:, 'name'].unique()]
                    }
    return data_meta

def assemble_target(data, feature='cumStatpoints'):
    targets = []
    for player_name in data.loc[:, 'name'].unique():
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        target = player_df.loc[:, feature].values.tolist()
        targets.append(target)
    return targets

def run_model(data_train,
              data_meta,
              num_epochs = 50,
              lr=1e-3,
              batch_size=64,
              save_path="data/models/"):
    estimator = DeepAREstimator(freq=data_meta['freq'],
                                prediction_length=data_meta['prediction_length'],
                                trainer=Trainer(batch_size=batch_size,
                                                epochs=num_epochs,
                                                learning_rate=lr,
                                                ctx='cpu',
                                                hybridize=False))
    predictor = estimator.train(data_train)
    predictor.serialize(Path(save_path))
    return predictor
