import json
import streamlit as st
import numpy as np 
import pandas as pd
from pandas.io.json import json_normalize
import requests as rqsts  



## HERE YE BE DRAGONS
# The following functions are either not complete or broken.
def generate_games_df(schedule_df):
    games_df = pd.DataFrame()
    games_series = schedule_df.games
    st.dataframe(games_series)
    for day in games_series:
        for game in day:
            if game['gameType'] == 'R': 
                st.dataframe(game)
                game_id = game['gamePk']
                game_teams = game['teams']
                away_team = game_teams['away']
                home_team = game_teams['home']
                st.write(home_team.keys())
                break
                # games_df.append(game, ignore_index=True)
    st.dataframe(games_df)
    return

@st.cache
def get_season_schedule(seasons_df=get_seasons(), season_id=20182019):
    season_series = seasons_df.loc['{}'.format(season_id), :]
    st.table(season_series)
    season_start = season_series['regularSeasonStartDate']
    season_end = season_series['regularSeasonEndDate']
    st.text('Season ID: {0}\nSeason start: {1}\nSeason end: {2}'.format(season_id, season_start, season_end))
    schedule_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/schedule?startDate={0}&endDate={1}'.format(season_start, season_end))
    if schedule_response.status_code != 200:
        st.error('Failed attempt to get full season schedule.')
        return
    schedule = schedule_response.content
    schedule_df = pd.read_json(schedule)
    schedule_df = json_normalize(schedule_df.dates)
    schedule_df.set_index('date', inplace=True)
    # schedule_df = json_normalize(schedule_df.games)
    # st.dataframe(schedule_df.games)
    generate_games_df(schedule_df)
    return schedule_df