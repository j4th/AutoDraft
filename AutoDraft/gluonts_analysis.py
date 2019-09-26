"""
This script is for analysing the outputs from the implementation of DeepAR in GluonTS
"""
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
import autodraft.visualization as viz
import autodraft.gluonts as glu

# @st.cache
def load_model(file_path):
    model = Predictor.deserialize(Path(file_path))
    return model

@st.cache
def get_data(path='data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def load_predictions(path='/home/ubuntu/AutoDraft/data/deepar_truncated_results_default.csv'):
    data = pd.read_csv(path, index_col=2)
    return data

@st.cache
def process_data(data):
    train_data, test_data, num_unique, targets = glu.prep_df(data,
                                                             column_list=['name', 'date', 'gameNumber', 'cumStatpoints'],
                                                             streamlit=True)
    return train_data, test_data, num_unique, targets

# @st.cache
def run_prediction(model, data):
    predictions = model.predict(dataset=data)
    return list(predictions)

def process_prediction(prediction):
    mean = prediction.mean_ts
    mean = mean.reset_index()
    mean = mean.rename(columns={'index': 'predictions'})
    mean = mean.reset_index()
    mean = mean.rename(columns={'index': 'gameNumber'})
    mean = mean.drop(columns=[0])
    mean.loc[:, 'gameNumber'] = mean.loc[:, 'gameNumber'] + 1
    conf = pd.DataFrame()
    conf.loc[:, 'low'] = prediction.quantile('0.05')
    conf.loc[:, 'high'] = prediction.quantile('0.95')
    full_df = pd.concat([mean, conf], axis=1)
    return full_df

def generate_prediction_df(predictions, train_data, test_data, drop=True, target='cumStatpoints'):
    names = test_data.loc[:, 'name'].unique()
    full_predictions = pd.DataFrame()
    for prediction, name in zip(predictions, names):
        player_df = pd.DataFrame()
        player_test_data = test_data.loc[test_data.loc[:, 'name'] == name].loc[:, ['name', 'date', target]]
        player_train_data = train_data.loc[train_data.loc[:, 'name'] == name].loc[:, ['name', 'date', target]]
        player_test_data.loc[:, 'date'] = pd.to_datetime(player_test_data.loc[:, 'date'])
        player_train_data.loc[:, 'date'] = pd.to_datetime(player_train_data.loc[:, 'date'])
        test_length = player_test_data.shape[0]
        prediction_df = process_prediction(prediction)
        # prediction_df.loc[:, 'name'] = name
        if drop:
            prediction_df = prediction_df.iloc[:test_length, :]
        player_test_data.reset_index(drop=True, inplace=True)
        prediction_df.reset_index(drop=True, inplace=True)
        player_train_data.reset_index(drop=True, inplace=True)
        player_test_df = pd.concat([player_test_data, prediction_df], axis=1)
        player_df = pd.concat([player_train_data, player_test_df])
        player_df.set_index('date', drop=False, inplace=True)
        full_predictions = pd.concat([full_predictions, player_df])
    return full_predictions

def main():
    # st.write('Loading model...')
    # model_path = "/home/ubuntu/AutoDraft/data/models/deepar_ne100_lre-4_bs64"
    # model = load_model(model_path)
    # st.write(type(model))
    # st.write('Done!')

    # st.write('Loading data...')
    # data_ts = get_data()
    # st.write('Done!')
    # train_data, test_data, num_unique, targets = process_data(data_ts)
    # data_meta = glu.generate_metadata(train_data, test_data, num_unique, index='gameNumber')

    # prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
    #                       for name in test_data.loc[:, 'name'].unique()]

    # train_ds = ListDataset([{FieldName.TARGET: target[:-prediction_length],
    #                          FieldName.START: start}
    #                          for target, start, prediction_length in zip(targets,
    #                                                                      data_meta['start'],
    #                                                                      prediction_lengths
    #                                                                     )],
    #                          freq=data_meta['freq']
    #                       )

    # st.write('Predicting...')
    # predictions = run_prediction(model, train_ds)
    # st.write('Done!')
    # prediction = process_prediction(predictions[0])
    # print(prediction.head())
    # predictions = generate_prediction_df(predictions, train_data, test_data)
    # st.dataframe(predictions.loc[predictions.loc[:, 'name'] == 'Leon Draisaitl'])
    # print(predictions.head())

    # save = st.checkbox('Save the predictions?')
    # while not save:
    #     pass

    # predictions.to_csv('/home/ubuntu/AutoDraft/data/deepar_truncated_results_ne100_lre-4_bs64.csv')

    st.write('Loading predictions...')
    predictions = load_predictions()
    st.write('Done!')
    # print(predictions.head())
    # st.dataframe(predictions)

    test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

    viz.plot_actual_predictions_series(predictions,
                                       None,
                                       model='deepar',
                                       target='cumStatpoints',
                                       metric='Rmse',
                                       player_name=test_player)

main()
