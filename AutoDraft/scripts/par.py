"""
This script is for generating points above replacement base values
"""
import streamlit as st
import pandas as pd 
import autodraft.visualization as viz

@st.cache
def get_data(path='../../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def calculate_means(data):
    par_df = pd.DataFrame()
    for date in data.loc[:, 'date'].unique():
        date_points = data.loc[data.loc[:, 'date'] == date].loc[:, ['cumStatpoints', 'statPoints']]
        mean_points = date_points.loc[:, 'statPoints'].mean()
        mean_cum = date_points.loc[:, 'cumStatpoints'].mean()
        date_df = pd.DataFrame({'date': date,
                                'meanStatpoints': mean_points,
                                'meanCumstatpoints': mean_cum},
                                index=['date'])
        par_df = pd.concat([par_df, date_df])
    par_df = par_df.sort_values(['date'])
    par_df.loc[:, 'cumStatpoints'] = par_df.loc[:, 'meanStatpoints'].cumsum() * (328 / par_df.shape[0])
    par_df = par_df.drop('meanStatpoints', axis=1)
    
    return par_df

def main():
    data = get_data()
    st.dataframe(data.head(10))

    par_df = calculate_means(data)
    st.dataframe(par_df)
    # st.write(par_df.shape[0])
    # # par_df.to_csv('/home/ubuntu/AutoDraft/data/joe_schmo_4_seasons.csv')

    # # test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

    # viz.plot_actual_predictions_series(None,
    #                                    None,
    #                                    joe=par_df,
    #                                    model='test',
    #                                    target='cumStatpoints',
    #                                    metric='Rmse',
    #                                    player_name='test')

main()
