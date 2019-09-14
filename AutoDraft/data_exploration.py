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

st.write("Let's grab the 4 most recent, COMPLETE, seasons.")
seasons_df = seasons_df.iloc[-5:-1, :]
st.dataframe(seasons_df)
st.write('Now on to getting our team rosters.')

@st.cache
def get_teams():
    teams_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/teams')
    if teams_response.status_code != 200:
        st.error('Failed attempt to get list of teams.')
        return
    teams = teams_response.content
    teams_df = pd.read_json(teams)
    teams_df = json_normalize(teams_df.teams)
    return teams_df

st.dataframe(get_teams())
st.write('From here, we will get the rosters for each team in the seasons we want.')

@st.cache
def get_roster(team_id=22, season_id=20182019):
    roster_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/teams/{0}?expand=team.roster&season={1}'.format(team_id, season_id))
    if roster_response.status_code != 200:
        st.error("Failed attempt to get team {0}'s roster for season {1}.".format(team_id, season_id))
        return
    roster = roster_response.content
    roster_df = pd.read_json(roster)
    roster_df = json_normalize(roster_df.teams)
    # st.dataframe(roster_df)
    # st.write(roster_df.columns)
    roster_list = roster_df['roster.roster'][0]
    roster_df = pd.DataFrame()
    for _, person in enumerate(roster_list):
        # st.write(person)
        person_info = person['person']
        person_position = person['position']
        player_dict = {'name': person_info['fullName'],
                        'position': person_position['code']}
        player_df = pd.DataFrame(player_dict, index=[person_info['id']])
        # st.dataframe(player_df)20182019
        roster_df = roster_df.append(player_df)
    # st.dataframe(roster_df)
    return roster_df

roster_team = st.text_input('Enter a team ID from the table above:', '22')
roster_year = st.text_input('Enter a season ID (eg. 20182019 for the 2018-2019 season)', '20182019')
st.write("Here is that team's roster for that year:")
st.dataframe(get_roster(team_id=int(roster_team), season_id=int(roster_year)))




