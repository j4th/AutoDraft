"""
This script is for exploring the implementation of DeepAR in GluonTS
"""
import json, itertools, os
import streamlit as st
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.transform import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from sklearn import preprocessing
from pathlib import Path

@st.cache
def get_data(path='data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def clean_duplicates(data, streamlit=False):
    unique = data.loc[:, 'name'].unique()
    num_unique = len(unique)
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        player_df = player_df.drop_duplicates(subset='date')
        clean = pd.concat([clean, player_df])
    return clean, num_unique

@st.cache
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

@st.cache
def clean_df(data, split_from='2018-10-03', streamlit=False):
    clean = clean_rookies_retirees(data, split_from=split_from)
    clean, num_unique = clean_duplicates(clean, streamlit=streamlit)
    return clean, num_unique

@st.cache
def subset_df(data, column_list=None):
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    data = data.loc[:, column_list]
    return data

@st.cache
def split_train_test(data, split_from='2018-10-03'):
    train = data.loc[data.loc[:, 'date'] < split_from]
    test = data.loc[data.loc[:, 'date'] >= split_from]
    return train, test

@st.cache
def prep_df(data, split_from='2018-10-03', column_list=None, streamlit=False):
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    data, num_unique = clean_df(data, split_from=split_from, streamlit=streamlit)
    targets, targets_meta = assemble_target(data)
    train, test = split_train_test(data)
    train = train.loc[:, column_list]
    test = test.loc[:, column_list]
    return train, test, num_unique, targets, targets_meta

@st.cache
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

@st.cache
def assemble_target(data, feature='cumStatpoints', stand=True, scale=True):
    targets = []
    targets_meta = pd.DataFrame()
    if stand:
        standardizer = preprocessing.StandardScaler()
    if scale:
        scaler = preprocessing.MaxAbsScaler()
    for player_name in data.loc[:, 'name'].unique():
        meta_dict = {'name':player_name}
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        if not stand and not scale:
            target = player_df.loc[:, feature].values.tolist()
        else:
            target = player_df.loc[:, feature].values.reshape(-1, 1)
            if stand:
                target = standardizer.fit_transform(target)
                meta_dict['mean'] = standardizer.mean_
                meta_dict['std'] = standardizer.scale_
            if scale:
                target = scaler.fit_transform(target)
                meta_dict['maxabs'] = scaler.scale_
            target = target.tolist()
            target = list(itertools.chain.from_iterable(target))
        targets.append(target)
        if stand and scale:
            meta = pd.DataFrame(meta_dict)
            targets_meta = pd.concat([targets_meta, meta])
    targets_meta = targets_meta.reset_index(drop=True)
    return targets, targets_meta

# @st.cache
def run_model(data_train,
              data_meta,
              num_epochs=50,
              lr=1e-3,
              batch_size=64,
              scaling=False,
              save_path="data/models/"):
    estimator = DeepAREstimator(freq=data_meta['freq'],
                                prediction_length=data_meta['prediction_length'],
                                scaling=scaling,
                                trainer=Trainer(batch_size=batch_size,
                                                epochs=num_epochs,
                                                learning_rate=lr,
                                                ctx='cpu',
                                                hybridize=False))
    predictor = estimator.train(data_train)
    predictor.serialize(Path(save_path))
    return predictor

def main():
    data_ts = get_data('data/full_dataset_4_seasons.csv')
    train_data, test_data, num_unique, targets, targets_meta = prep_df(data_ts, column_list=['name', 'gameNumber', 'cumStatpoints'], streamlit=True)
    st.write(targets)
    st.dataframe(targets_meta)
    st.write('Number of unique players: {}'.format(num_unique))
    st.write('Train dataset shape: {0}\nTest dataset shape: {1}'.format(train_data.shape, test_data.shape))
    st.write('Training dataframe head:')
    st.dataframe(train_data.head())
    data_meta = generate_metadata(train_data, test_data, num_unique, index='gameNumber')
    st.write(data_meta)
    prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
                          for name in test_data.loc[:, 'name'].unique()]

    train_ds = ListDataset([{FieldName.TARGET: target[:-prediction_length],
                             FieldName.START: start}
                             for target, start, prediction_length in zip(targets,
                                                                         data_meta['start'],
                                                                         prediction_lengths
                                                                        )],
                             freq=data_meta['freq']
                          )

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start}
                            for target, start in zip(targets,
                                                     data_meta['start']
                                                    )],
                            freq=data_meta['freq']
                         )

    # train = st.checkbox('Train the model?')
    # while not train:
    #     pass

    folder_path = '/home/ubuntu/AutoDraft/data/models/deepar_ss_ne100_lre-4_bs64'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)        

    
    predictor = run_model(train_ds,
                          data_meta,
                          num_epochs=100,
                          lr=1e-4,
                          batch_size=64,
                          scaling=False,
                          save_path=folder_path)

    pd.to_pickle(targets_meta, folder_path+'/targets_meta.p')

main()
