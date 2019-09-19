import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
import NHL_API as nhl

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
def load_pickle(path='./data/temp/arima_results.p'):
    data = pd.read_pickle(path)
    return data

temp_results = load_pickle()
st.write('Current shape of ARIMA results: {}'.format(temp_results.shape))

data = load_csv()
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

autocorrelation_plot(smaller_test_player)
fig = plt.gcf()
st.pyplot(fig)

seasons = nhl.get_seasons()
st.dataframe(seasons)

model = ARIMA(smaller_test_player, order=(3,1,1))
model_fit = model.fit(disp=0)
st.text(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
fig = plt.gcf()
st.pyplot(fig)

residuals.plot(kind='kde')
fig = plt.gcf()
st.pyplot(fig)
st.write(residuals.describe())

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

stepwise_model = auto_arima(smaller_test_player, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

smaller_test_player_train = smaller_test_player[:'2018-10-03']
smaller_test_player_test = smaller_test_player['2018-10-03':]

stepwise_model.fit(smaller_test_player_train)
predictions = stepwise_model.predict(n_periods=smaller_test_player_test.shape[0])

test_residuals = calculate_test_residuals(predictions, smaller_test_player_test)
testMfe, testMae, testRmse, testResiduals = calculate_errors(test_residuals)

# @st.cache
def player_arima(data, player_name='Leon Draisaitl',index='date' ,feature='cumStatpoints' , player_id=None, roster=None, p=3, d=1, q=1, summary=False):
    if player_id and type(roster) != None: # TODO: add logic for if the player ID is given but not a roster (use function in package)
        player_name = roster[roster['Unnamed: 0'] == player_id]
    player_df = data[data['name'] == player_name]
    player_train_df = player_df[player_df['date'] < '2018-10-03']
    player_test_df = player_df[player_df['date'] >= '2018-10-03']
    player_train_df = player_train_df.loc[:, [index, feature]]
    player_train_df = player_train_df.set_index(index, drop=True)
    player_test_df = player_test_df.loc[:, [index, feature]]
    player_test_df = player_test_df.set_index(index, drop=True)
    # st.write('{} train'.format(player_name))
    # st.dataframe(player_train_df)
    # st.write('{} test'.format(player_name))
    # st.dataframe(player_test_df)
    player_train_df = player_train_df[:'2018-10-03']
    if player_train_df.shape[0] == 0:
        st.write('{} is a rookie!'.format(player_name))
        return None
    player_test_df = player_test_df['2018-10-03':]
    if player_test_df.shape[0] == 0:
        st.write('{} retired!'.format(player_name))
        return None
    # try:
    #     model = ARIMA(player_df, order=(p,d,q))
    #     model_fit = model.fit(disp=0)
    # except ValueError:
    #     return None
    try:
        try:
            model = auto_arima(player_test_df, start_p=1, start_q=1,
                                max_p=3, max_q=3, m=7,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
            aic = model.aic()
            model.fit(player_train_df)
        except IndexError:
            st.write('Index error for {}'.format(player_name))
            return None
        except:
            st.write('Unhandled error for {}'.format(player_name))
            return None
    except ValueError:
        st.write("{} doesn't have enough data!".format(player_name))
        return None
    predictions = model.predict(n_periods=player_test_df.shape[0])
    prediction_residuals = calculate_test_residuals(predictions, player_test_df)
    if summary: st.text(model_fit.summary())
    train_residuals = pd.DataFrame(model.resid())
    trainMfe, trainMae, trainRmse, trainResiduals = calculate_errors(train_residuals)
    testMfe, testMae, testRmse, testResiduals = calculate_errors(prediction_residuals)
    # try:
    #     ar_coeffs = model.arparams()
    #     arCoeff1, arCoeff2, arCoeff3 = unpack_coeffs(*ar_coeffs) # TODO: handle multiple coefficients
    # except AttributeError:
    #     arCoeff1, arCoeff2, arCoeff3 = (None, None, None)
    # try:
    #     ma_coeffs = model.maparams()
    #     maCoeff1, maCoeff2, maCoeff3 = unpack_coeffs(*ma_coeffs) # TODO: handle multiple coefficients
    # except AttributeError:
    #     maCoeff1, maCoeff2, maCoeff3 = (None, None, None)
    # if q == 1:
    #     maCoeff = model_fit.maparams.item()
    #     results_df = pd.DataFrame({'trainMfe':mfe,
    #                             'trainMae':mae,
    #                             'trainRmse':rmse,
    #                             'arCoeff1':arCoeff1,
    #                             'arCoeff2':arCoeff2,
    #                             'arCoeff3':arCoeff3,
    #                             'maCoeff':maCoeff,
    #                             'trainResiduals':[residuals]}, index=[player_name])
    # elif q == 2:
    #     maCoeffs = model_fit.maparams.tolist()
    #     maCoeff1, maCoeff2 = unpack_coeffs(*maCoeffs)
    #     results_df = pd.DataFrame({'trainMfe':mfe,
    #                             'trainMae':mae,
    #                             'trainRmse':rmse,
    #                             'arCoeff1':arCoeff1,
    #                             'arCoeff2':arCoeff2,
    #                             'arCoeff3':arCoeff3,
    #                             'maCoeff1':maCoeff1,
    #                             'maCoeff2':maCoeff2,
    #                             'trainResiduals':[residuals]}, index=[player_name])
    results_df = pd.DataFrame({'trainMfe':trainMfe,
                                'trainMae':trainMae,
                                'trainRmse':trainRmse,
                                'testMfe':testMfe,
                                'testMae':testMae,
                                'testRmse':testRmse,
                                # 'arCoeff1':arCoeff1,
                                # 'arCoeff2':arCoeff2,
                                # 'arCoeff3':arCoeff3,
                                # 'maCoeff1':maCoeff1,
                                # 'maCoeff2':maCoeff2,
                                # 'maCoeff3':maCoeff3,
                                'trainResiduals':[train_residuals],
                                'testResiduals':[prediction_residuals]}, index=[player_name])
    return results_df

arima_response = player_arima(data)
st.dataframe(arima_response)

# @st.cache
def all_player_arima(data, roster):
    results = pd.DataFrame()
    for index, player in roster.iterrows():
        player_name = player['name']
        player_results = player_arima(data, player_name=player_name)
        if type(player_results) is type(None):
            st.write('Skipping {}'.format(player_name))
            continue
        st.dataframe(player_results)
        results = pd.concat([results, player_results])
        results.to_pickle('./data/temp/arima_results.p')
    return results

all_arima_results = all_player_arima(data, full_roster)