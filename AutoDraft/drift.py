"""
This incomplete function will calculate a drift model in-season for players as a baseline
"""
import pandas as pd
import streamlit as st
import autodraft.visualization as viz

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

def insample_drift(player_df, end_date='2018-10-03'):
    player_df = player_df.loc[player_df.loc[:, 'date'] < end_date]
    player_df = player_df.sort_values('date')
    player_series = player_df.loc[:, 'cumStatpoints']
    try:
        start_val = player_series.iloc[0]
    except IndexError as e:
        raise e
    player_drift = (player_series.iloc[-1] - start_val) / player_series.shape[0]
    drift_series = [start_val + i*player_drift for i in range(player_series.shape[0])]
    drift_df = pd.DataFrame({'date':player_df.loc[:, 'date'],
                             'name':player_df.loc[:, 'name'],
                             'cumStatpoints':player_series,
                             'predictions':drift_series})
    drift_df = drift_df.reset_index(drop=True)
    drift_df = drift_df.drop(0)
    return drift_df

def outsample_drift(player_df, start_date='2018-10-03'):
    player_sample = player_df.loc[player_df.loc[:, 'date'] < start_date]
    player_df = player_df.loc[player_df.loc[:, 'date'] >= start_date]
    player_df = player_df.sort_values('date')
    player_df = player_df.drop_duplicates('date')
    player_series = player_df.loc[:, 'cumStatpoints']
    try:
        sample_start = player_sample['cumStatpoints'].iloc[0]
        # start_val = player_series.iloc[0]
    except IndexError as e:
        raise e
    sample_end = player_sample['cumStatpoints'].iloc[-1]
    player_drift = (sample_end - sample_start) / player_sample.shape[0]
    drift_series = [sample_end + (i+1)*player_drift for i in range(player_series.shape[0])]
    drift_df = pd.DataFrame({'date':player_df.loc[:, 'date'],
                             'name':player_df.loc[:, 'name'],
                             'cumStatpoints':player_series,
                             'predictions':drift_series})
    drift_df = drift_df.reset_index(drop=True)
    return drift_df

@st.cache
def generate_drift_df(data, location='out'):
    output_df = pd.DataFrame()
    for player in data.loc[:, 'name'].unique():
        player_df = data.loc[data.loc[:, 'name'] == player]
        if location == 'in':
            try:
                player_drift = insample_drift(player_df)
            except IndexError:
                continue
        else:
            try:
                player_drift = outsample_drift(player_df)
            except IndexError:
                continue
        output_df = pd.concat([output_df, player_drift])
    return output_df

@st.cache
def generate_drift_errors(data, location='out'):
    output_df = pd.DataFrame()
    for player in data.loc[:, 'name'].unique():
        player_df = data.loc[data.loc[:, 'name'] == player]
        if location == 'in':
            try:
                player_drift = insample_drift(player_df)
            except IndexError:
                continue
        else:
            try:
                player_drift = outsample_drift(player_df)
            except IndexError:
                continue
        if location == 'in':
            mfe, mae, rmse = viz.calculate_errors(viz.calculate_residuals(player_drift,
                                                                        start_date=None,
                                                                        end_date='2018-10-03'),
                                                    start_date=None,
                                                    end_date='2018-10-03')
        else:
            try:
                start = player_drift['date'].values[0]
            except IndexError:
                continue
            mfe, mae, rmse = viz.calculate_errors(viz.calculate_residuals(player_drift,
                                                                        start_date=start,
                                                                        end_date=None),
                                                    start_date=start,
                                                    end_date=None)
        drift_df = pd.DataFrame({'name':player,
                                 'mfe':mfe,
                                 'mae':mae,
                                 'rmse':rmse},
                                index=[player])
        drift_df = drift_df.reset_index(drop=True)
        output_df = pd.concat([output_df, drift_df])
    return output_df


def main():
    data = load_csv()
    st.dataframe(data.head())
    test_player = st.text_input('Player to work with: ', 'Leon Draisaitl')
    test_drift = outsample_drift(data.loc[data.loc[:, 'name'] == test_player])
    mfe, mae, mse = viz.calculate_errors(viz.calculate_residuals(test_drift, start_date=test_drift['date'].values[0], end_date=None), start_date=test_drift['date'].values[0], end_date=None)
    st.write(mfe, mae, mse)

    # drift_errors = generate_drift_errors(data)
    # st.dataframe(drift_errors)

    # drift_errors.to_csv('/home/ubuntu/AutoDraft/data/drift_results.csv')

    drift_df = generate_drift_df(data)
    st.dataframe(drift_df)

    # drift_df.to_csv('/home/ubuntu/AutoDraft/data/drift_out_results.csv')

    # mfe, mae, mse = viz.calculate_errors(viz.calculate_residuals(drift_df, start_date=None, end_date='2018-10-03'), start_date=None, end_date='2018-10-03')
    # st.write(mfe, mae, mse)

main()
