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

@st.cache
def load_pickle(path='./data/arima_results.p'):
    data = pd.read_pickle(path)
    return data

data = load_csv()
results = load_pickle()

st.write('Data shape: {0}\nResults shape: {1}'.format(data.shape, results.shape))