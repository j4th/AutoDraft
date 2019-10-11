"""
This is the streamlit web-app
"""
import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
import autodraft.visualization as viz
import autodraft.gluonts as glu

# @st.cache
def load_arima(path='../data/output/arima_results_m3.p'):
    data = pd.read_pickle(path)
    return data

@st.cache
def get_data(path='../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def load_nn(path='../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    data = pd.read_csv(path, index_col=2)
    model_name = path.split('/')[-1].split('.')[0]
    return data, model_name

@st.cache
def load_future(path='../data/output/deepar_20192020_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def load_joe(path='../data/input/joe_schmo_4_seasons.csv'):
    joe = pd.read_csv(path)
    return joe

@st.cache
def load_leaderboard(path='../data/output/leaderboard_bad.csv'):
    table = pd.read_csv(path)
    return table

@st.cache
def load_roster(path='../data/input/full_roster_4_seasons.csv'):
    roster = pd.read_csv(path)
    return roster

@st.cache
def get_error_df():
    error_df = viz.generate_error_df(['../data/output/drift_out_results.csv',
                                      '../data/output/arima_results_m3.p',
                                      '../data/output/deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv',
                                      '../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv',
                                      '../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl82_cl3.csv'],
                                     '../data/input/full_dataset_4_seasons.csv',
                                     '../data/output/drift_out_results.csv')
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

@st.cache
def assemble_diagnoses(data, errors, roster, model='deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3'):
    position_map = {'C': 1, 'L': 2, 'R': 3, 'D': 4}
    model_names_index = 'name.'+model
    error_df = errors[[model_names_index, model]]
    error_df = error_df.rename(mapper={'{}'.format(model_names_index): 'name',
                                       '{}'.format(model): 'error'},
                               axis=1)
    output_df = pd.DataFrame()
    for player_name in error_df['name'].unique():
        player_df = data[data['name'] == player_name]
        player_df = player_df.drop_duplicates('date')
        player_position = roster[roster['name'] == player_name]['position']
        try:
            player_position = player_position.values[0]
        except IndexError:
            continue
        player_position = position_map[player_position]
        player_error = error_df[error_df['name'] == player_name]['error'].values[0]
        train_df = player_df[player_df['date'] < '2018-10-03']
        train_size = train_df.shape[0]
        test_df = player_df[player_df['date'] >= '2018-10-03']
        test_size = test_df.shape[0]
        player_output = pd.DataFrame({'name': player_name,
                                      'error': player_error,
                                      'trainSize': train_size,
                                      'testSize': test_size,
                                      'position': player_position},
                                      index=[player_name])
        output_df = pd.concat([output_df, player_output])
    output_df['errorBinned'], error_bins = pd.cut(output_df['error'], 10, duplicates='drop', retbins=True)
    output_df['trainBinned'], train_bins = pd.cut(output_df['trainSize'], 10, duplicates='drop', retbins=True)
    output_df['testBinned'], test_bins = pd.cut(output_df['testSize'], 10, duplicates='drop', retbins=True)
    return output_df, error_bins, train_bins, test_bins

# @st.cache
# def dfScatter(df, xcol='trainSize', ycol='error', catcol='testSize'):
#     fig, ax = plt.subplots()
#     categories = np.unique(df[catcol])
#     colors = np.linspace(0, 1, len(categories))
#     colordict = dict(zip(categories, colors))  

#     df["Color"] = df[catcol].apply(lambda x: colordict[x])
#     ax.scatter(df[xcol], df[ycol], c=df.Color)
#     return fig

def main():
    predictions_deepar, deepar_name = load_nn('../data/output/deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv')
    predictions_mv, mv_name = load_nn('../data/output/deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv')
    predictions_arima = load_arima()
    predictions_future = load_future()
    leaderboard = process_future(predictions_future)
    player_data = get_data()
    joe = load_joe()
    roster = load_roster()
    # leaderboard = load_leaderboard()
    errors = get_error_df()
    # st.dataframe(player_data[player_data['name'] == 'Denis Malgin'])
    # st.dataframe(errors)
    # arima_values = full_arima(player_data, predictions_arima)
    # nn_values = predictions_deepar.copy()


    diagnose_arima, arima_error_bins, arima_train_bins, arima_test_bins = assemble_diagnoses(player_data, errors, roster, model='arima_results_m3')
    diagnose_nn, nn_error_bins, nn_train_bins, nn_test_bins = assemble_diagnoses(player_data, errors, roster, model='deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3')
    diagnose_mv, mv_error_bins, mv_train_bins, mv_test_bins = assemble_diagnoses(player_data, errors, roster, model='deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3')

    st.title('AutoDraft')
    st.write('Welcome to AutoDraft! The intention of this app '
             'is to give predictive insights into which players '
             'will either under or over perform in the coming season, and WHEN. ')

    st.header('Individual Player Predictions')
    st.write('Please feel free to change the model that is being used '
             'for the forecasts (the methodologies used for each model '
             'are included below). Select a player from the dropdown to '
             'change the player being predicted. Please feel free to interact '
             'with the legend and the sliding window at the bottom.')

    predict_choice = st.sidebar.checkbox('Predict 2019-2020 season?')
    # predict_choice = False
    
    if not predict_choice:
        model_choice = st.sidebar.selectbox('Please select the model to view '
                                    'projections for: ', ['ARIMA', 'DeepAR', 'MV-DeepAR'], index=2)
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

    st.header('2019/2020 Leaderboards')
    st.write('Here is the leaderboard for cumulative points, '
             'forecasted by DeepAR, for the coming season. '
             'Though there are numerical stability issues for '
             'forecasting into the coming season, the relative '
             'rankings seem to not only be plausible, '
             'but somewhat in line with other expert sources. '
             'Feel free to change the amount of players included in '
             'the leaderboard by adjusting it on the sidebar.')

    head_length = st.sidebar.selectbox('Select number of places in the upcoming season leaderboard to view: ', [5, 10, 15, 20], 0)

    st.dataframe(leaderboard['name'].head(int(head_length)))

    # viz.ridge_plots(errors[['arima_results_m3', 'deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3', 'deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3']])
    # st.write(errors['arima_results_m3'].median())
    # st.write(errors['deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3'].median())
    # st.write(errors['deepar_truncated_results_mv_unit_s_ne300_lr1e-3_bs64_nl3_cl3'].median())

    # st.text(viz.test_errors(errors,
    #                         dist1='arima_results_m3',
    #                         dist2='deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3',
    #                         report=True))

    # train_errors = []
    # train_bins = []
    # for game_bin in diagnose_arima['errorBinned'].unique():
    #     st.write(game_bin)
    #     error = diagnose_arima[diagnose_arima['errorBinned'].isin(range(int(game_bin.left), int(game_bin.right)))]['error']
    #     error = error.sum()
    #     train_errors.append(error)
    #     train_bins.append(train_bins)
    # st.write(train_errors)
    # st.write(train_bins)
    # st.write(arima_error_bins)

    # plt.clf()
    # plot = plt.bar(x=arima_train_bins, height=train_errors)
    # plot = plt.gcf()
    # st.pyplot(plot)

    # plot = diagnose_arima['testBinned'].value_counts(sort=False).plot.bar(rot=25, figsize=(14, 8))
    # plt.rcParams.update({'font.size': 30})
    # plot.set_xlabel('Games in Training Set')
    # plot.axes.get_xaxis().set_visible(False)
    # plot.set_ylim(0, 175)
    # plot.set_ylabel('Count')
    # plot = plt.gcf()
    # st.pyplot(plot, figsize=(14, 8))   

    # plot = diagnose_arima['testBinned'].value_counts(sort=False).plot.bar(rot=25, figsize=(10,8))
    # plot.set_xlabel('Games in Test Set')
    # plot.set_ylabel('Count')
    # plot = plt.gcf()
    # st.pyplot(plot)

    # plot = diagnose_mv.plot.scatter(x='trainSize', y='error', alpha=0.3)
    # plot = plt.gcf()
    # st.pyplot(plot)

    # plot = diagnose_mv.plot.scatter(x='testSize', y='error', alpha=0.3)
    # plot = plt.gcf()
    # st.pyplot(plot)

    # plt.clf()
    # plot = plt.scatter(x=diagnose_mv['testSize'], y=diagnose_mv['error'], c=diagnose_mv['trainSize'])
    # plt.xlabel('Games Played in Test Set')
    # plt.ylim(0,20)
    # plt.ylabel('MASE')
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Games Played in Training Set')
    # plot = plt.gcf()
    # st.pyplot(plot)

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
