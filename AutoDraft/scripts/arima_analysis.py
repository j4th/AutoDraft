"""
Script for analyzing ARIMA results using Streamlit.
"""
import pandas as pd
import streamlit as st
import autodraft.visualization as viz

@st.cache
def load_csv(path='../../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache(ignore_hash=True)
def load_results(path='../../data/output/arima_results_m3.p'):
    data = pd.read_pickle(path)
    data.loc[:, 'name'] = data.index
    data.drop_duplicates('name', inplace=True)
    return data

@st.cache(ignore_hash=True)
def load_transformed_results(path='../../data/output/arima_results_m3_yj.p'):
    data = pd.read_pickle(path)
    data.loc[:, 'name'] = data.index
    data.drop_duplicates('name', inplace=True)
    return data

def main():
    data = load_csv()
    results = load_results()
    results_yj = load_transformed_results('../../data/output/arima_results_m3_yj.p')

    st.text('Stats data shape: {0}\nARIMA results shape: {1}'.format(data.shape, results.shape))
    # st.write('Stat dataframe head:')
    # st.dataframe(data.head())
    # st.write('ARIMA results dataframe:')
    # st.dataframe(results)
    # st.write('Yeo-Johnsoned ARIMA results dataframe:')
    # st.dataframe(results_yj)

    # plot_hists(['testRmse'])

    test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

    # test_intervals = return_intervals()

    transform = st.checkbox('Use transformed (YJ) data?')
    if not transform:
        viz.plot_actual_predictions_series(data, results, player_name=test_player)
    else:
        viz.plot_actual_predictions_series(data, results_yj, player_name=test_player)

main()
