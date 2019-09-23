import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool
from bokeh.models.glyphs import Patch
from bokeh.plotting import figure, show
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import NHL_API as nhl

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache(ignore_hash=True)
def load_results(path='./data/arima_results_m3.p'):
    data = pd.read_pickle(path)
    data.loc[:,'name'] = data.index
    data.drop_duplicates('name', inplace=True)
    return data

@st.cache(ignore_hash=True)
def load_transformed_results(path='./data/arima_results_m3_yj.p'):
    data = pd.read_pickle(path)
    data.loc[:,'name'] = data.index
    data.drop_duplicates('name', inplace=True)
    return data

data = load_csv()
results = load_results()
results_yj = load_transformed_results('./data/arima_results_m3_yj.p')

st.text('Stats data shape: {0}\nARIMA results shape: {1}'.format(data.shape, results.shape))
st.write('Stat dataframe head:')
st.dataframe(data.head())
st.write('ARIMA results dataframe:')
st.dataframe(results)
st.write('Yeo-Johnsoned ARIMA results dataframe:')
st.dataframe(results_yj)

# plot_hists(['testRmse'])

test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

# test_intervals = return_intervals()

transform = st.checkbox('Use transformed (YJ) data?')
if not transform:
    nhl.plot_actual_predictions_series(data, results, player_name=test_player)
else:
    nhl.plot_actual_predictions_series(data, results_yj, player_name=test_player)