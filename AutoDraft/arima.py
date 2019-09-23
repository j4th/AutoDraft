import time, copy
import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from sklearn.preprocessing import PowerTransformer
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
# from scipy.special import boxcox1p
# from scipy.special import inv_boxcox1p
import NHL_API as nhl

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
def load_pickle(path='./data/temp/arima_results.p'):
    data = pd.read_pickle(path)
    return data

# temp_results = load_pickle()
# st.write('Current shape of ARIMA results: {}'.format(temp_results.shape))

data = copy.deepcopy(load_csv())
data['date'] = pd.to_datetime(data['date'])
full_roster = load_csv('./data/full_roster_4_seasons.csv')
st.write('Number of players captured: {}'.format(full_roster.shape[0]))
st.dataframe(full_roster)
test_player = data[data['name'] == 'Leon Draisaitl']
# test_player.set_index('date', inplace=True, drop=False)
st.dataframe(test_player)

test_player.plot(x='date', y='cumStatpoints')
fig = plt.gcf()
st.pyplot(fig)

small_test_player = test_player.loc[:, ['date', 'cumStatpoints']]
smaller_test_player = small_test_player.set_index('date', drop=True)
st.dataframe(smaller_test_player)

# decomposition_df = smaller_test_player
# decomposition_df.loc[:, 'date'] = decomposition_df.index
# st.dataframe(decomposition_df)
# decomposition = seasonal_decompose(smaller_test_player, model='additive')
# decomposition.plot()
# fig = plt.gcf()
# st.pyplot(fig)

# autocorrelation_plot(smaller_test_player)
# fig = plt.gcf()
# st.pyplot(fig)

@st.cache
def calculate_errors(residuals):
    num_residuals = len(residuals)
    mfe = (residuals.sum() / num_residuals).tolist()[0]
    mae = (residuals.abs().sum() / num_residuals).tolist()[0]
    rmse = (residuals.pow(2).sum().pow(0.5)).tolist()[0]
    residuals = residuals.values
    residuals = [value.item() for value in residuals]
    return mfe, mae, rmse, residuals

@st.cache
def unpack_coeffs(coeff1=None, coeff2=None, coeff3=None):
    # if coeff3:
    return coeff1, coeff2, coeff3
    # return coeff1, coeff2

@st.cache
def get_predictions(data, model_fit):
    predictions = model_fit.predict('2018-10-03', '2019-04-06', typ='levels')
    real = data[data['date'] >= '2018-10-03'].cumStatpoints.to_numpy()
    st.dataframe(predictions)
    st.dataframe(real)
    return

@st.cache
def calculate_test_residuals(prediction_array, test_data):
    prediction_array = prediction_array.reshape(len(test_data), 1)
    test_data = test_data.values
    residuals = np.subtract(test_data, prediction_array)
    residuals = residuals.tolist()
    residuals = pd.DataFrame(residuals)
    return residuals

# train_residuals.plot()
# fig = plt.gcf()
# st.pyplot(fig)

# train_residuals.plot(kind='kde')
# fig = plt.gcf()
# st.pyplot(fig)

# test_residuals.plot()
# fig = plt.gcf()
# st.pyplot(fig)

# test_residuals.plot(kind='kde')
# fig = plt.gcf()
# st.pyplot(fig)

# @st.cache
def player_arima(data, player_name, index='date' ,feature='cumStatpoints' , forecast_from='2018-10-03', transform='yj', player_id=None, roster=None, summary=False):
    if player_id and type(roster) != None: # TODO: add logic for if the player ID is given but not a roster (use function in package)
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
        player_train_df.loc[:, 'logValues'] = np.log(player_train_df['cumStatpoints']) # TODO: make this stat agnostic
    elif transform == 'yj':
        transformer = PowerTransformer()
        transformer.fit(player_train_df.values.reshape(-1, 1))
        player_train_df.loc[:, 'transformedValues'] = transformer.transform(player_train_df['cumStatpoints'].values.reshape(-1, 1))
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
        model = pm.auto_arima(player_train_df, start_p=1, start_q=1,
                                max_p=5, max_q=5, max_d=3, m=3,
                                start_P=0, start_Q=0, seasonal=True,
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
        low_intervals = transformer.inverse_transform(intervals[:,0].reshape(-1, 1))
        high_intervals = transformer.inverse_transform(intervals[:,1].reshape(-1, 1))
    end_time = time.time()
    if transform != 'yj':
        low_intervals = []
        high_intervals = []
        for low, high in intervals:
            low_intervals.append(low)
            high_intervals.append(high)
    prediction_residuals = calculate_test_residuals(predictions, player_test_df)
    if summary: st.text(model.summary())
    train_residuals = pd.DataFrame(model.resid())
    trainMfe, trainMae, trainRmse, trainResiduals = calculate_errors(train_residuals)
    testMfe, testMae, testRmse, testResiduals = calculate_errors(prediction_residuals)
    model_params = model.get_params()
    p, d, q = model_params['order']
    try:
        P, D, Q, m = model_params['seasonal_order']
    except TypeError:
        st.write('Search failed to find valid options.')
        return None
    st.write("{0}'s Auto-ARIMA({1},{2},{3})({4},{5},{6},{7}) took {8:.3f} seconds.".format(player_name, p, d, q, P, D, Q, m, end_time-start_time))
    results_df = pd.DataFrame({'forecastStart':forecast_from,
                                'aic':model.aic(),
                                'p':p,
                                'd':d,
                                'q':q,
                                'P':P,
                                'D':D,
                                'Q':Q,
                                'm':m,
                                'trainMfe':trainMfe, # TODO: store more info (eg. confidence intervals, )
                                'trainMae':trainMae,
                                'trainRmse':trainRmse,
                                'trainResiduals':[train_residuals],
                                'testMfe':testMfe,
                                'testMae':testMae,
                                'testRmse':testRmse,
                                'testResiduals':[prediction_residuals], 
                                'intervalLow':[low_intervals],
                                'intervalHigh':[high_intervals]}, index=[player_name])
    return results_df

# arima_response = player_arima(data, player_name='Leon Draisaitl', transform='yj', summary=True).values.tolist()
# st.dataframe(arima_response)

# @st.cache
def all_player_arima(data, roster, transform='none', print_status=False):
    if print_status: print('Running Auto-ARIMAs...')
    results = pd.DataFrame()
    for index, player in roster.iterrows():
        if print_status: print('Player {}'.format(index))
        player_name = player['name']
        player_results = player_arima(data, player_name=player_name, transform=transform)
        if type(player_results) is type(None):
            st.write('Skipping {}'.format(player_name))
            continue
        st.dataframe(player_results)
        results = pd.concat([results, player_results])
        results.to_pickle('./data/temp/arima_results_TEMP.p')
    if print_status: print('Done!')
    return results

all_arima_results = nhl.all_player_arima(data, roster=full_roster, transform='yj', print_status=False)
