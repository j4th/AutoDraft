"""
Module for running ARIMA on players
"""
import time
import streamlit as st
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import PowerTransformer

def calculate_errors(residuals):
    """ Calculates errors based on residuals """
    num_residuals = len(residuals)
    mfe = (residuals.sum() / num_residuals).tolist()[0]
    mae = (residuals.abs().sum() / num_residuals).tolist()[0]
    rmse = (residuals.pow(2).sum().pow(0.5)).tolist()[0]
    residuals = residuals.values
    residuals = [value.item() for value in residuals]
    return mfe, mae, rmse

def calculate_test_residuals(prediction_array, test_data):
    """ Calculates test residuals based on prediction and test data """
    prediction_array = prediction_array.reshape(len(test_data), 1)
    test_data = test_data.values
    residuals = np.subtract(test_data, prediction_array)
    residuals = residuals.tolist()
    residuals = pd.DataFrame(residuals)
    return residuals

def player_arima(data,
                 player_name,
                 index='date',
                 feature='cumStatpoints',
                 forecast_from='2018-10-03',
                 transform='none',
                 player_id=None,
                 roster=None,
                 summary=False):
    """ performs Auto-ARIMA on a single player """
    # TODO: add logic for if the player ID is given but not a roster (use function in package)
    if player_id and roster:
        player_name = roster[roster['Unnamed: 0'] == player_id]
    player_df = data[data['name'] == player_name]
    player_df.drop_duplicates(subset='date', keep='first', inplace=True)
    player_train_df = player_df[player_df['date'] < forecast_from]
    player_test_df = player_df[player_df['date'] >= forecast_from]
    player_train_df = player_train_df.loc[:, [index, feature]]
    player_train_df = player_train_df.set_index(index, drop=True)
    if player_train_df.shape[0] == 0:
        st.write('{} is a rookie!'.format(player_name))
        return None
    if transform == 'log':
        # TODO: make this stat agnostic
        player_train_df.loc[:, 'logValues'] = np.log(player_train_df['cumStatpoints'])
    elif transform == 'yj':
        transformer = PowerTransformer()
        transformer.fit(player_train_df.values.reshape(-1, 1))
        player_train_df.loc[:, 'transformedValues'] = transformer \
                                                      .transform(
                                                          player_train_df['cumStatpoints'] \
                                                          .values.reshape(-1, 1))
        player_train_df.drop('cumStatpoints', axis=1, inplace=True)
    player_test_df = player_test_df.loc[:, [index, feature]]
    player_test_df = player_test_df.set_index(index, drop=True)
    # player_train_df = player_train_df[:'2018-10-03']
    # player_test_df = player_test_df['2018-10-03':]
    if player_test_df.shape[0] == 0:
        st.write('{} retired!'.format(player_name))
        return None
    start_time = time.time()
    st.write('Searching ARIMA parameters for {}...'.format(player_name))
    try:
        model = pm.auto_arima(player_train_df,
                              start_p=1,
                              start_q=1,
                              max_p=5,
                              max_q=5,
                              max_d=3,
                              m=3,
                              start_P=0,
                              start_Q=0,
                              seasonal=True,
                              information_criterion='aicc',
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
        st.write('Model built, fitting...')
        model.fit(player_train_df)
    except ValueError:
        st.write("{} doesn't have enough data!".format(player_name))
        return None
    except IndexError:
        st.write('Index error for {}'.format(player_name))
        return None
    except:
        st.write('Unhandled error for {}'.format(player_name))
        return None
    predictions, intervals = model.predict(n_periods=player_test_df.shape[0], return_conf_int=True)
    if transform == 'log':
        predictions = np.exp(predictions)
        intervals = np.exp(intervals)
    elif transform == 'yj':
        predictions = transformer.inverse_transform(predictions.reshape(-1, 1))
        low_intervals = transformer.inverse_transform(intervals[:, 0].reshape(-1, 1))
        high_intervals = transformer.inverse_transform(intervals[:, 1].reshape(-1, 1))
    end_time = time.time()
    if transform != 'yj':
        low_intervals = []
        high_intervals = []
        for low, high in intervals:
            low_intervals.append(low)
            high_intervals.append(high)
    prediction_residuals = calculate_test_residuals(predictions, player_test_df)
    if summary:
        st.text(model.summary())
    train_residuals = pd.DataFrame(model.resid())
    train_mfe, train_mae, train_rmse = calculate_errors(train_residuals)
    test_mfe, test_mae, test_rmse = calculate_errors(prediction_residuals)
    model_params = model.get_params()
    p, d, q = model_params['order']
    try:
        P, D, Q, m = model_params['seasonal_order']
    except TypeError:
        st.write('Search failed to find valid options.')
        return None
    st.write("{0}'s Auto-ARIMA({1},{2},{3})({4},{5},{6},{7}) took {8:.3f} seconds." \
             .format(player_name, p, d, q, P, D, Q, m, end_time-start_time))
    results_df = pd.DataFrame({'forecastStart':forecast_from,
                               'aic':model.aic(),
                               'p':p,
                               'd':d,
                               'q':q,
                               'P':P,
                               'D':D,
                               'Q':Q,
                               'm':m,
                               'trainMfe':train_mfe,
                               'trainMae':train_mae,
                               'trainRmse':train_rmse,
                               'trainResiduals':[train_residuals],
                               'testMfe':test_mfe,
                               'testMae':test_mae,
                               'testRmse':test_rmse,
                               'testResiduals':[prediction_residuals],
                               'intervalLow':[low_intervals],
                               'intervalHigh':[high_intervals]},
                              index=[player_name])
    return results_df

def all_player_arima(data, roster, save_loc, transform='none', print_status=False):
    """ performs Auto_ARIMA on all players in a given roster """
    if print_status:
        print('Running Auto-ARIMAs...')
    results = pd.DataFrame()
    for index, player in roster.iterrows():
        if print_status:
            print('Player {}'.format(index))
        player_name = player['name']
        player_results = player_arima(data, player_name=player_name, transform=transform)
        if isinstance(player_results, type(None)):
            st.write('Skipping {}'.format(player_name))
            continue
        st.dataframe(player_results)
        results = pd.concat([results, player_results])
        results.to_pickle(save_loc)
    if print_status:
        print('Done!')
    return results
