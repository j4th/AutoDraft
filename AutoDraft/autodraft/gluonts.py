"""
Module for working with GluonTS and models from it
"""
import itertools
from pathlib import Path
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import numpy as np

# def clean_duplicates(data, streamlit=False):
#     unique = data.loc[:, 'name'].unique()
#     num_unique = len(unique)
#     clean = pd.DataFrame()
#     for player in unique:
#         player_df = data.loc[data.loc[:, 'name'] == player]
#         player_df = player_df.drop_duplicates(subset='date')
#         clean = pd.concat([clean, player_df])
#     return clean, num_unique

def clean_duplicates(data, streamlit=False):
    unique = data.loc[:, 'name'].unique()
    clean = pd.DataFrame()
    for player in unique:
        player_df = data.loc[data.loc[:, 'name'] == player]
        player_df = player_df.drop_duplicates(subset='date')
        clean = pd.concat([clean, player_df])
    return clean

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

# def clean_df(data, split_from='2018-10-03', streamlit=False):
#     clean = clean_rookies_retirees(data, split_from=split_from)
#     clean, num_unique = clean_duplicates(clean, streamlit=streamlit)
#     return clean, num_unique

@st.cache
def clean_df(data, split_from='2018-10-03', streamlit=False):
    clean = clean_rookies_retirees(data, split_from=split_from)
    clean = clean_duplicates(clean, streamlit=streamlit)
    clean = remove_small_ts(clean, split_from)
    return clean

def sort_dates(data):
    data_df = pd.DataFrame()
    for player_name in data['name'].unique():
        player_df = data[data['name'] == player_name]
        player_df = player_df.sort_values('date')
        data_df = pd.concat([data_df, player_df])
    return data_df

def remove_small_ts(data, split_from='2018-10-03'):
    output_df = pd.DataFrame()
    st.write(data.keys())
    for player_name in data['name'].unique():
        player_df = data[data['name'] == player_name]
        player_test = player_df[player_df['date'] >= split_from]
        player_df = player_df[player_df['date'] < split_from]
        if player_df.shape[0] < 82:
            continue
        player_df = pd.concat([player_df, player_test])
        output_df = pd.concat([output_df, player_df])
    return output_df

def subset_df(data, column_list=None):
    if column_list is None:
        column_list = ['name', 'date', 'cumStatpoints']
    data = data.loc[:, column_list]
    return data

def split_train_test(data, split_from='2018-10-03'):
    train = data.loc[data.loc[:, 'date'] < split_from]
    test = data.loc[data.loc[:, 'date'] >= split_from]
    return train, test

# def prep_df(data, split_from='2018-10-03', column_list=None, streamlit=False, scale=True, target_output_df=False):
#     if column_list is None:
#         column_list = ['name', 'date', 'cumStatpoints']
#     data, num_unique = clean_df(data, split_from=split_from, streamlit=streamlit)
#     if target_output_df:
#         targets, targets_meta, target_df = assemble_target(data, scale=scale, df_output=target_output_df)
#     else:
#         targets, targets_meta = assemble_target(data, scale=scale, df_output=target_output_df)
#     train, test = split_train_test(data)
#     train = train.loc[:, column_list]
#     test = test.loc[:, column_list]
#     if target_output_df:
#         return train, test, num_unique, targets, targets_meta, target_df
#     return train, test, num_unique, targets, targets_meta

def prep_df(data, roster, split_from='2018-10-03', column_list=None, streamlit=False, scale=True, multivar=True):
    st.write('Prepping DF...')
    # st.dataframe(data.head())
    data = data.sort_values('date')
    if column_list is None:
        column_list = ['name', 'date', 'gameNumber', 'cumStatpoints']
    data = clean_df(data, split_from=split_from, streamlit=streamlit)
    data = sort_dates(data)
    # st.dataframe(data.head())
    if multivar:
        # data = remove_small_ts(data, split_from)
        stat_cat_features = assemble_static_cat(data, roster)
        dyn_cat_features = assemble_dynamic_cat(data, roster)
        dyn_real_features, dyn_real_features_meta = days_since_last_game(data)
    targets, targets_meta = assemble_target(data, scale=scale)
    train, test = split_train_test(data)
    # st.dataframe(train.head())
    st.write(column_list)
    train = train.loc[:, column_list]
    # st.dataframe(train.head())
    test = test.loc[:, column_list]
    return train, test, targets, targets_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta

def encode_roster(feature_df, roster):
    position_map = {'C': 1, 'L': 2, 'R': 3, 'D': 4}
    roster = roster.replace({'position': position_map})
    # st.dataframe(roster)
    output_df = pd.DataFrame()
    for player_name in feature_df['name'].unique():
        player_df = feature_df[feature_df['name'] == player_name]
        player_position = roster[roster['name'] == player_name]['position'].iloc[0]
        player_df['position'] = player_position
        output_df = pd.concat([output_df, player_df])
    return output_df

def assemble_dynamic_cat(data, roster, feature_list=['teamId', 'opponentId'], position=False):
    features = data.copy()
    output = []
    features = features.loc[:, ['name', 'date'] + feature_list]
    # features = encode_roster(features, roster)
    for player_name in features['name'].unique():
        player_df = features[features['name'] == player_name]
        # player_df = player_df[['position'] + feature_list]
        player_df = player_df[feature_list]
        output.append(player_df.values.reshape(2, -1))
    return output

def assemble_static_cat(data, roster):
    output = []
    position_map = {'C': 1, 'L': 2, 'R': 3, 'D': 4}
    roster = roster.replace({'position': position_map})
    for player_name in data['name'].unique():
        position = roster[roster['name'] == player_name]['position'].values.item(0)
        output.append([position])
    return output

def days_since_last_game(data, scale=True):
    date_df = data[['date', 'name']]
    date_df['date'] = pd.to_datetime(date_df['date'])
    output = []
    if scale:
        output_meta = pd.DataFrame()
    for player_name in date_df['name'].unique():
        player_df = date_df[date_df['name'] == player_name]
        date_diff = player_df['date']
        date_diff = date_diff.diff()
        date_diff.iloc[0] = pd.Timedelta('187 days 00:00:00')
        date_diff = date_diff.dt.days.values.reshape(-1, 1)
        if scale:
            meta_dict = {'name': player_name}
            scaler = preprocessing.MinMaxScaler()
            date_diff = scaler.fit_transform(date_diff)
            meta_dict['min'] = scaler.min_
            meta_dict['scale'] = scaler.scale_
            meta = pd.DataFrame(meta_dict)
            output_meta = pd.concat([output_meta, meta])
            date_diff = date_diff.tolist()
            # st.write(len(target))
            date_diff = np.array(list(itertools.chain.from_iterable(date_diff))).reshape(1, -1)
            # st.write(target)
        # player_df['dslg'] = date_diff
        output.append(date_diff)
    return output, output_meta

# def generate_metadata(train_data, test_data, num_unique, index=None):
#     # prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
#     #                                     for name in test_data.loc[:, 'name'].unique()]
#     if index is None:
#         if 'date' in train_data.columns:
#             index = 'date'
#         elif 'gameNumber' in train_data.columns:
#             index = 'gameNumber'
#     if index == 'date':
#         data_meta = {'num_series': num_unique,
#                     'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
#                                 .shape[0] for name in train_data.loc[:, 'name'].unique()],
#                     'prediction_length': 82,
#                     'freq': '1D',
#                     'start': [pd.Timestamp(train_data.loc[train_data.loc[:, 'name'] == name] \
#                                             .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
#                                             .index[0], 'date'], freq='1D')
#                             for name in train_data.loc[:, 'name'].unique()]
#                     }
#     elif index == 'gameNumber':
#         data_meta = {'num_series': num_unique,
#                     'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
#                                   .shape[0] for name in train_data.loc[:, 'name'].unique()],
#                     'prediction_length': 82,
#                     'freq': '1D',
#                     'start': [train_data.loc[train_data.loc[:, 'name'] == name] \
#                                         .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
#                                         .index[0], 'gameNumber']
#                               for name in train_data.loc[:, 'name'].unique()]
#                     }
#     return data_meta

def generate_metadata(train_data, test_data, index=None):
    if index is None:
        if 'date' in train_data.columns:
            index = 'date'
        elif 'gameNumber' in train_data.columns:
            index = 'gameNumber'
    prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
                         for name in test_data.loc[:, 'name'].unique()]
    num_unique = len(train_data['name'].unique())
    if index == 'date':
        data_meta = {'num_series': num_unique,
                    'num_steps': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                .shape[0] for name in train_data.loc[:, 'name'].unique()],
                    'prediction_length': prediction_lengths,
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
                    'prediction_length': prediction_lengths,
                    'freq': '1D',
                    'start': [train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .loc[train_data.loc[train_data.loc[:, 'name'] == name] \
                                        .index[0], 'gameNumber']
                              for name in train_data.loc[:, 'name'].unique()]
                    }
    return data_meta

# def assemble_target(data, feature='cumStatpoints', stand=False, scale=True, df_output=False):
#     targets = []
#     targets_meta = pd.DataFrame()
#     # if stand:
#     #     standardizer = preprocessing.StandardScaler()
#     # if scale:
#     #     scaler = preprocessing.MinMaxScaler()
#     if df_output:
#         targets_df = pd.DataFrame()
#     for player_name in data.loc[:, 'name'].unique():
#         meta_dict = {'name':player_name}
#         player_df = data.loc[data.loc[:, 'name'] == player_name]
#         if not stand and not scale:
#             target = player_df.loc[:, feature].values.tolist()
#         else:
#             target = player_df.loc[:, feature].values.reshape(-1, 1)
#             if stand:
#                 standardizer = preprocessing.StandardScaler()
#                 target = standardizer.fit_transform(target)
#                 meta_dict['mean'] = standardizer.mean_
#                 meta_dict['std'] = standardizer.scale_
#             if scale:
#                 scaler = preprocessing.MinMaxScaler()
#                 target = scaler.fit_transform(target)
#                 meta_dict['min'] = scaler.min_
#                 meta_dict['scale'] = scaler.scale_
#             target = target.tolist()
#             target = list(itertools.chain.from_iterable(target))
#         if df_output:
#             target_df = pd.DataFrame({'name':player_name,
#                                       'gameNumber': player_df.loc[:, 'gameNumber'],
#                                       feature: target})
#             targets_df = pd.concat([targets_df, target_df])
#         targets.append(target)
#         if stand or scale:
#             meta = pd.DataFrame(meta_dict)
#             targets_meta = pd.concat([targets_meta, meta])
#     targets_meta = targets_meta.reset_index(drop=True)
#     if df_output:
#         return targets, targets_meta, targets_df
#     return targets, targets_meta

def assemble_target(data, feature='cumStatpoints', stand=False, scale=True):
    targets = []
    targets_meta = pd.DataFrame()
    # if stand:
    #     standardizer = preprocessing.StandardScaler()
    # if scale:
    #     scaler = preprocessing.MinMaxScaler()
    for player_name in data.loc[:, 'name'].unique():
        meta_dict = {'name':player_name}
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        if not stand and not scale:
            target = player_df.loc[:, feature].values.tolist()
        else:
            target = player_df.loc[:, feature].values.reshape(-1, 1)
            if stand:
                standardizer = preprocessing.StandardScaler()
                target = standardizer.fit_transform(target)
                meta_dict['mean'] = standardizer.mean_
                meta_dict['std'] = standardizer.scale_
            if scale:
                scaler = preprocessing.MinMaxScaler()
                target = scaler.fit_transform(target)
                meta_dict['min'] = scaler.min_
                meta_dict['scale'] = scaler.scale_
            target = target.tolist()
            # st.write(len(target))
            target = list(itertools.chain.from_iterable(target))
            # st.write(target)
        targets.append(target)
        if stand or scale:
            meta = pd.DataFrame(meta_dict)
            targets_meta = pd.concat([targets_meta, meta])
    targets_meta = targets_meta.reset_index(drop=True)
    return targets, targets_meta

def run_model(data_train,
              data_meta,
              save_path,
              num_epochs = 50,
              lr=1e-3,
              batch_size=64):
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
