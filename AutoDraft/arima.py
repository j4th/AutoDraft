import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from NHL_API import *

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

data = load_csv()
full_roster = load_csv('./data/full_roster_4_seasons.csv')
st.write('Number of players captured: {}'.format(full_roster.shape[0]))
st.dataframe(full_roster)
test_player = data[data['name'] == 'Leon Draisaitl']
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
    mfe = (residuals.sum() / num_residuals).to_numpy().item()
    mae = (residuals.abs().sum() / num_residuals).to_numpy().item()
    rmse = (residuals.pow(2).sum().pow(0.5)).to_numpy().item()
    residuals = residuals.to_numpy()
    residuals = [value.item() for value in residuals]
    return mfe, mae, rmse, residuals

@st.cache
def unpack_coeffs(coeff1, coeff2, coeff3):
    return coeff1, coeff2, coeff3

# @st.cache
def player_arima(data, player_name='Leon Draisaitl',index='date' ,feature='cumStatpoints' , player_id=None, roster=None, p=3, d=1, q=1, summary=False):
    if player_id and type(roster) != None: # TODO: add logic for if the player ID is given but not a roster (use function in package)
        player_name = roster[roster['Unnamed: 0'] == player_id]
    player_df = data[data['name'] == player_name]
    player_df = player_df.loc[:, [index, feature]]
    player_df = player_df.set_index(index, drop=True)
    model = ARIMA(player_df, order=(p,d,q))
    model_fit = model.fit(disp=0)
    if summary: st.text(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    mfe, mae, rmse, residuals = calculate_errors(residuals)
    ar_coeffs = model_fit.arparams.tolist()
    arCoeff1, arCoeff2, arCoeff3 = unpack_coeffs(*ar_coeffs) # TODO: handle multiple coefficients
    maCoeff = model_fit.maparams.item()
    results_df = pd.DataFrame({'mfe':mfe,
                                'mae':mae,
                                'rmse':rmse,
                                'arCoeff1':arCoeff1,
                                'arCoeff2':arCoeff2,
                                'arCoeff3':arCoeff3,
                                'maCoeff':maCoeff,
                                'residuals':[residuals]}, index=[player_name])
    return results_df

arima_response = player_arima(data)
st.dataframe(arima_response)