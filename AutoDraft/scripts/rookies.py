"""
Script for finding rookies, and evaluating model performance for them.
"""
import json
import requests as rq 
import streamlit as st 
import pandas as pd 
import numpy as np
from pandas.io.json import json_normalize

@st.cache
def get_data(path='../../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

@st.cache
def get_tournaments(key):
    global_tournament_list_response = rq.get('https://api.sportradar.us/hockey-t1/ice/en/tournaments.json?api_key={}'.format(key))
    global_tournament_df = json_normalize(pd.read_json(global_tournament_list_response.content)['tournaments'])
    return global_tournament_df

@st.cache
def get_tournament_info(key):
    tournament_info = rq.get('https://api.sportradar.us/hockey-t1/ice/en/tournaments/sr:tournament:261/seasons.json?api_key={}'.format(key))
    tournament_info = json.loads(tournament_info.content)
    tournament_info = pd.DataFrame(tournament_info['seasons'])
    return tournament_info

@st.cache
def get_tournament_details(key):
    tournament_sched = rq.get('https://api.sportradar.us/hockey-t1/ice/en/tournaments/sr:season:33357/schedule.json?api_key={}'.format(key))
    tournament_sched.raise_for_status()
    tournament_sched = json.loads(tournament_sched.content)
    tournament_sched = pd.DataFrame(tournament_sched['sport_events'])
    st.dataframe(tournament_sched)
    tournament_games = tournament_sched['id']
    tournament_competitors = pd.DataFrame()
    for match in tournament_sched.loc[:, 'competitors']:
        team1 = match[0]
        competitor = pd.DataFrame(team1, index=[0])
        tournament_competitors = pd.concat([tournament_competitors, competitor])
    tournament_teams = pd.DataFrame({'id': tournament_competitors['id'].unique()})
    tournament_teams['name'] = tournament_competitors['name'].unique()
    return tournament_teams, tournament_games

def main():
    data = get_data()
    st.write('Number of rookies: {}'.format(len(data.loc[:, 'name'].unique()) - (len(data.loc[data.loc[:, 'date'] < '2018-10-03'].loc[:, 'name'].unique()))))
    
    all_players = set(data.loc[:, 'name'].unique())
    pro_players = set(data.loc[data.loc[:, 'date'] < '2018-10-03'].loc[:, 'name'].unique())
    rookie_players = list(all_players - pro_players)

    rookies = data.loc[data.loc[:, 'name'].isin(rookie_players)]
    st.dataframe(rookies)

    test_rookies_dict = {'Rasmus Dahlin': 'SHL', 'Miro Heiskanen': 'Liiga'}

    test_rookies = rookies.loc[rookies.loc[:, 'name'].isin(test_rookies_dict.keys())]
    st.dataframe(test_rookies)

    global_hockey_API_key = open('./keys/sportsradar-global-ice-hockey.txt', mode='r').readline().strip()
    global_tournament_df = get_tournaments(global_hockey_API_key)
    global_tournament_df = global_tournament_df.loc[global_tournament_df.loc[:, 'name'].isin(test_rookies_dict.values())]
    st.dataframe(global_tournament_df)

    tournament_info = get_tournament_info(global_hockey_API_key)
    st.dataframe(tournament_info)

    tournament_teams, tournament_games = get_tournament_details(global_hockey_API_key)
    st.dataframe(tournament_teams)
    st.write(tournament_games)

    # team_profile = rq.get('https://api.sportradar.us/hockey-t1/ice/en/teams/sr:competitor:3734/profile.json?api_key={}'.format(global_hockey_API_key))
    # team_profile.raise_for_status()
    # # st.write(team_profile.content)
    # team_profile = json.loads(team_profile.content)
    # st.json(team_profile)

    game_lineup = rq.get('https://api.sportradar.us/hockey-t1/ice/en/matches/sr:match:9655945/lineups.json?api_key={}'.format(global_hockey_API_key))
    game_lineup.raise_for_status()
    game_lineup = json.loads(game_lineup.content)
    st.json(game_lineup)

main()
