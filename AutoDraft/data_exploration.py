import json 
import streamlit as st
import numpy as np 
import pandas as pd
from pandas.io.json import json_normalize
import requests as rqsts  

st.title('NHL API Dataset Assembly')
st.write('This report will cover assembling a player-level game-by-game dataset for 4 NHL seasons.')
st.write('For the most complete dataset, we will go through each game on each day so as to get all stats, but only for those players that played in the game.')

@st.cache
def get_seasons():
    seasons_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/seasons')
    if seasons_response.status_code != 200:
        st.error('Failed attempt to get list of seasons.')
        return
    seasons = seasons_response.content
    seasons_df = pd.read_json(seasons)
    seasons_df = json_normalize(seasons_df.seasons)
    seasons_df.set_index('seasonId', inplace=True)
    return seasons_df

st.write('Here is the full dataframe of NHL seasons with available stats:')
seasons_df = get_seasons()
st.dataframe(seasons_df)



