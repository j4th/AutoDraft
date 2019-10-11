"""
This script is for analysing the outputs from the implementation of DeepAR in GluonTS
"""
import os, time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
import autodraft.visualization as viz
import autodraft.gluonts as glu
import autodraft.api as nhl
from bokeh.sampledata.perceptions import probly


# @st.cache
def load_model(file_path):
    model = Predictor.deserialize(Path(file_path))
    return model

@st.cache
def get_data(path='../../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
# def load_predictions(path='/home/ubuntu/AutoDraft/data/deepar_truncated_results_ne100_lre-4_bs64.csv'):
#     data = pd.read_csv(path, index_col=2)
#     return data

@st.cache
def load_predictions(path='../../data/output/deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    data = pd.read_csv(path, index_col=2)
    model_name = path.split('/')[-1].split('.')[0]
    return data, model_name

@st.cache
def load_joe():
    joe = pd.read_csv('../../data/input/joe_schmo_4_seasons.csv')
    return joe

@st.cache
def get_roster(path='../../data/input/full_roster_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def process_data(data, roster):
    train, test, targets, targets_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta = glu.prep_df(data,
                                                                                                                                        roster,
                                                                                                                                        column_list=['name', 'date', 'gameNumber', 'cumStatpoints'],
                                                                                                                                        streamlit=True,
                                                                                                                                        scale=True)
    return train, test, targets, targets_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta

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

def generate_prediction_df(predictions, train_data, test_data, drop=True, target='cumStatpoints', scaled=None, scaling_loc=None):
    if scaled is not None:
        scaling_meta = pd.read_pickle(scaling_loc)
        st.write(scaling_meta)
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
        if scaled == 'ss':
            scale_data = scaling_meta.loc[scaling_meta.loc[:, 'name'] == name]
            for column in ['predictions', 'low', 'high']:
                prediction_df.loc[:, column] = ((prediction_df.loc[:, column] * scale_data['maxabs']) \
                                               * scale_data['std']) + scale_data['mean']
        elif scaled == 'unit':
            scale_data = scaling_meta.loc[scaling_meta.loc[:, 'name'] == name]
            for column in ['predictions', 'low', 'high']:
                prediction_df.loc[:, column] = (prediction_df.loc[:, column] - scale_data['min'].values) / scale_data['scale'].values
        player_train_data.reset_index(drop=True, inplace=True)
        player_test_df = pd.concat([player_test_data, prediction_df], axis=1)
        player_df = pd.concat([player_train_data, player_test_df])
        player_df.set_index('date', drop=False, inplace=True)
        full_predictions = pd.concat([full_predictions, player_df])
    return full_predictions

@st.cache
def generate_future(predictions, test_data, scaled='unit', scaling_loc=None):
    if scaled is not None:
        scaling_meta = pd.read_pickle(scaling_loc)
    futures = predictions.copy()
    names = test_data['name'].unique()
    future_df = pd.DataFrame()
    for name, future in zip(names, futures):
        # player_start = test_data[test_data['name'] == name]['cumStatpoints'].tolist()[-1]
        player_future = process_prediction(future)
        if scaled == 'unit':
            scale_data = scaling_meta.loc[scaling_meta.loc[:, 'name'] == name]
            for column in ['predictions', 'low', 'high']:
                player_future.loc[:, column] = ((player_future.loc[:, column] - scale_data['min'].values) / scale_data['scale'].values)
        for column in ['predictions', 'low', 'high']:
            player_future.loc[:, column] = player_future.loc[:, column]
        player_future['name'] = name
        future_df = pd.concat([future_df, player_future])
    st.dataframe(future_df)
    return future_df

@st.cache
def get_error_df():
    error_df = viz.generate_error_df(['../../data/output/drift_out_results.csv',
                                      '../../data/output/arima_results_m3.p',
                                      '../../data/output/deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv',
                                      '../../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv',
                                      '../../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl82_cl3.csv'],
                                     '../../data/input/full_dataset_4_seasons.csv',
                                     '../../data/output/drift_out_results.csv')
    return error_df

@st.cache
def get_model_error_df(predictions):
    model_error_df = viz.calculate_error_df(predictions, '../../data/output/drift_out_results.csv', start_date='2018-10-03', end_date=None, use_mase=True)
    return model_error_df

# @st.cache
def get_future_covars(data):
    _, season_start, season_end = nhl.get_current_season()
    # st.write(season_start.values, season_end.values)
    schedule = nhl.get_schedule(season_start.values[0], season_end.values[0])
    covars = schedule[['gameDate', 'teams.home.team.id', 'teams.away.team.id']]
    covars = covars.rename({'gameDate': 'date',
                            'teams.home.team.id': 'teamId',
                            'teams.away.team.id': 'opponentId'}, axis=1)
    st.dataframe(covars)
    output_df = pd.DataFrame()
    for player_name in data['name'].unique():
        player_df = data[data['name'] == player_name]
        player_team = player_df['teamId'].values[0]
        player_schedule = covars[covars['teamId'] == player_team]
        player_schedule['name'] = player_name
        dslg, dslg_meta = glu.days_since_last_game(player_schedule)
        st.write(player_schedule.shape)
        st.dataframe(pd.DataFrame(dslg[0]))
        player_schedule['dslg'] = dslg
        output_df = pd.concat([output_df, player_schedule])
        st.dataframe(output_df)
        return
    return output_df

def main():
    st.write('Loading model...')
    model_path = "../../data/models/deepar_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3_ed16"
    model = load_model(model_path)
    st.write(type(model))
    st.write('Done!')

    st.write('Loading data...')
    data_ts = get_data()
    roster = get_roster()
    st.write('Done!')

    get_future_covars(data_ts)
    return

    train_data, test_data, targets, targets_meta, stat_cat_features, dyn_cat_features, dyn_real_features, dyn_real_features_meta = process_data(data_ts, roster)
    data_meta = glu.generate_metadata(train_data, test_data, index='gameNumber')

    # st.dataframe(train_data.head())

    # prediction_lengths = [test_data.loc[test_data.loc[:, 'name'] == name].shape[0]
    #                       for name in test_data.loc[:, 'name'].unique()]

    # input_list = [{FieldName.TARGET: target[:-prediction_length],
    #                 FieldName.START: start}
    #                 for target, start, prediction_length in zip(targets,
    #                                                             data_meta['start'],
    #                                                             data_meta['prediction_length']
    #                                                             )]    

    input_list = [{FieldName.TARGET: target[:-prediction_length],
                             FieldName.START: start,
                             FieldName.FEAT_STATIC_CAT: stat_cat,
                             FieldName.FEAT_DYNAMIC_CAT: dyn_cat,
                             FieldName.FEAT_DYNAMIC_REAL: dyn_real}
                             for target, start, prediction_length, stat_cat, dyn_cat, dyn_real in zip(targets,
                                                                                                        data_meta['start'],
                                                                                                        data_meta['prediction_length'],
                                                                                                        stat_cat_features,
                                                                                                        dyn_cat_features,
                                                                                                        dyn_real_features
                                                                                                        )]

    st.write(len(input_list))
    st.write(len(input_list[0]['target']))
    st.write(np.array(input_list[0]['target']).reshape(1, -1))
    st.write(len(input_list[0]['feat_static_cat']))
    st.write(np.array(input_list[0]['feat_static_cat']).reshape(1, -1))
    st.write(input_list[0]['feat_dynamic_cat'].shape)
    st.write(input_list[0]['feat_dynamic_cat'])
    st.write(input_list[0]['feat_dynamic_real'].shape)
    st.write(input_list[0]['feat_dynamic_real'])

    train_ds = ListDataset(input_list,
                             freq=data_meta['freq']
                          )

    st.write('Predicting...')
    start_time = time.time()
    predictions = run_prediction(model, train_ds)
    end_time = time.time()
    st.write('Done!')
    st.write(end_time - start_time)
    st.dataframe(predictions)
    # prediction = process_prediction(predictions[0])
    # st.dataframe(prediction)
    # print(prediction.head())

    # future = generate_future(predictions, train_data, scaled='unit', scaling_loc=model_path+'/targets_meta.p')
    # future.to_csv('/home/ubuntu/AutoDraft/data/deepar_20192020_results_s_ne300_lr1e-3_bs64_nl3_cl3.csv')

    predictions = generate_prediction_df(predictions, train_data, test_data, scaled='unit', scaling_loc=model_path+'/targets_meta.p')
    st.dataframe(predictions.head())
    print(predictions.head())

    # # save = st.checkbox('Save the predictions?')
    # # while not save:
    # #     pass

    # # predictions = predictions.apply(lambda x: str(x['date']), )
    # # st.dataframe(predictions)

    # predictions.to_csv('/home/ubuntu/AutoDraft/data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl82_cl3.csv')
    # # predictions.to_feather('/home/ubuntu/AutoDraft/data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.feather')

    # joe = load_joe()
    # # st.write('Loading predictions...')
    # # predictions, model_name = load_predictions('/home/ubuntu/AutoDraft/data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl82_cl3.csv')
    # # predictions = pd.read_feather('/home/ubuntu/AutoDraft/data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.feather')
    # st.dataframe(predictions.head())
    # st.write('Done!')

    # test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

    # errors = get_error_df()
    # st.dataframe(errors)

    # viz.plot_actual_predictions_series(predictions,
    #                                    None,
    #                                    errors,
    #                                    joe=joe,
    #                                    model='deepar',
    #                                    target='cumStatpoints',
    #                                    player_name=test_player,
    #                                    deepar_model_name=model_name)

    # error_df = get_error_df()
    # st.dataframe(error_df.head())

    # viz.ridge_plots(error_df)

    # st.text(viz.test_errors(error_df,
    #                          dist1='deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl82_cl3',
    #                          dist2='deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3',
    #                          report=True))

main()
