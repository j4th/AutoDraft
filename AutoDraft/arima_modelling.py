"""
Script for generating ARIMA models for players
"""
import copy
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import autodraft.arima as ar

@st.cache
def load_csv(path='../data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
def load_pickle(path='../data/temp/arima_results.p'):
    data = pd.read_pickle(path)
    return data

def main():
    # temp_results = load_pickle()
    # st.write('Current shape of ARIMA results: {}'.format(temp_results.shape))

    data = copy.deepcopy(load_csv())
    data['date'] = pd.to_datetime(data['date'])
    full_roster = load_csv('../data/full_roster_4_seasons.csv')
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

    # TODO: move this plotting functionality into a standalone function so it can be called (or not)
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

    # arima_response = ar.player_arima(data,
    #                                  player_name='Leon Draisaitl',
    #                                  transform='yj',
    #                                  summary=True).values.tolist()
    # st.dataframe(arima_response)

    arima_results = ar.player_arima(data, player_name='Leon Draisaitl', transform='none')
    st.dataframe(arima_results)
    # all_arima_results = ar.all_player_arima(data,
    #                                         roster=full_roster,
    #                                         transform='yj',
    #                                         print_status=False)
    # st.dataframe(all_arima_results)

main()
