"""
This is the streamlit web-app
"""
import streamlit as st
import os
from pathlib import Path
import pandas as pd
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
import autodraft.visualization as viz
import autodraft.gluonts as glu

# @st.cache
def load_arima(path='../data/arima_results_m3.p'):
    data = pd.read_pickle(path)
    return data

@st.cache
def get_data(path='../data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def load_nn(path='../data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    data = pd.read_csv(path, index_col=2)
    model_name = path.split('/')[-1].split('.')[0]
    return data, model_name

@st.cache
def load_future(path='../data/deepar_20192020_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def load_joe(path='../data/joe_schmo_4_seasons.csv'):
    joe = pd.read_csv(path)
    return joe

@st.cache
def load_leaderboard(path='../data/leaderboard_bad.csv'):
    table = pd.read_csv(path)
    return table

@st.cache
def get_error_df():
    error_df = viz.generate_error_df()
    return error_df

# @st.cache
def full_arima(player_data, predictions_arima):
    all_values = pd.DataFrame()
    for name in predictions_arima.index:
        try:
            arima_values = viz.calculate_predictions(player_data, predictions_arima, player_name=name, target='cumStatpoints')
        except AttributeError:
            continue
        else:
            arima_values.loc[:, 'name'] = name
        all_values = pd.concat([all_values, arima_values])
    return all_values

@st.cache
def process_future(predictions, subtract=True):
    future = predictions.copy()
    if subtract:
        calculated_df = pd.DataFrame()
        for player_name in future['name'].unique():
            player_df = future[future['name'] == player_name]
            player_df = player_df.sort_values('gameNumber')
            player_start = player_df['predictions'].iloc[0]
            player_df['predictions'] = player_df['predictions'] - player_start
            calculated_df = pd.concat([calculated_df, player_df])
        future = calculated_df
    future = future.sort_values('predictions', ascending=False)
    future = future.drop_duplicates('name')
    future = future.reset_index(drop=True).reset_index()
    future['index'] = future['index'] + 1
    future = future.set_index('index')
    return future

def main():
    predictions_deepar, deepar_name = load_nn('../data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv')
    predictions_mv, mv_name = load_nn('../data/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv')
    predictions_arima = load_arima()
    predictions_future = load_future()
    leaderboard = process_future(predictions_future)
    player_data = get_data()
    joe = load_joe()
    # leaderboard = load_leaderboard()
    errors = get_error_df()
    # arima_values = full_arima(player_data, predictions_arima)
    # nn_values = predictions_deepar.copy()

    st.title('AutoDraft')
    st.write('Welcome to AutoDraft! The intention of this app '
             'is to give predictive insights into which players '
             'will either under or over perform in the coming season, and WHEN. '
             'For quick insights, here is the table of the top 5 '
             'players by projected cumulative points for the coming season '
             'according to DeepAR (more on this below): ')

    head_length = st.sidebar.selectbox('Select number of places in the upcoming season leaderboard to view: ', [5, 10, 15, 20], 0)

    st.dataframe(leaderboard['name'].head(int(head_length)))

    st.header('Individual Player Predictions')
    st.write('Please feel free to change the model that is being used '
             'for the forecasts (the methodologies used for each model '
             'are included below). Select a player from the dropdown to '
             'change the player being predicted. Please feel free to interact '
             'with the legend and the sliding window at the bottom.')

    predict_choice = st.sidebar.checkbox('Predict 2019-2020 season?')
    
    if not predict_choice:
        model_choice = st.sidebar.selectbox('Please select the model to view '
                                    'projections for: ', ['ARIMA', 'DeepAR', 'MV-DeepAR'])
    else:
        st.write('Currently predictions are only available from the DeepAR model.')
        model_choice = 'DeepAR'
    player_choice = st.sidebar.selectbox('Please select the player to view '
                                         'projections for: ',
                                         player_data['name'].unique(),
                                         index=player_data['name'].unique().tolist().index('Leon Draisaitl'))

    if not predict_choice:
        if model_choice == 'ARIMA':
            viz.plot_actual_predictions_series(player_data,
                                            predictions_arima,
                                            errors,
                                            joe=joe,
                                            model='arima',
                                            target='cumStatpoints',
                                            player_name=player_choice)
        elif model_choice == 'DeepAR':
            viz.plot_actual_predictions_series(predictions_deepar,
                                            None,
                                            errors,
                                            joe=joe,
                                            model='deepar',
                                            target='cumStatpoints',
                                            player_name=player_choice,
                                            deepar_model_name=deepar_name)
        elif model_choice == 'MV-DeepAR':
            viz.plot_actual_predictions_series(predictions_mv,
                                            None,
                                            errors,
                                            joe=joe,
                                            model='deepar',
                                            target='cumStatpoints',
                                            player_name=player_choice,
                                            deepar_model_name=mv_name)
    else:
        viz.plot_actual_predictions_series(predictions_future,
                                            None,
                                            None,
                                            joe=joe,
                                            model='deepar',
                                            target='cumStatpoints',
                                            player_name=player_choice,
                                            deepar_model_name=deepar_name)

    viz.ridge_plots(errors)

    st.text(viz.test_errors(errors,
                            dist1='arima_results_m3',
                            dist2='deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3',
                            report=True))

    st.header('Methodology')
    st.write('Two different models are used for the predictions '
             'available in this app: ARIMA, and DeepAR. '
             'For both of them, the target is set to be the cumulative '
             'points over the course of a season, based on the players '
             'games over the course of the last 3 seasons. ')

    st.subheader('ARIMA')
    st.write('Lorum ipsum...')

    st.subheader('DeepAR')
    st.write('Lorum ipsum...')

main()
