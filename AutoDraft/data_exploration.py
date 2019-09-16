import json 
import streamlit as st
import numpy as np 
import pandas as pd
from pandas.io.json import json_normalize
import requests as rqsts
import plotly.express as px

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
        player_df = pd.DataFrame(player_dict, index=[person_info['id']]) ## TODO: should change to not extract as int64
        # st.dataframe(player_df)20182019
        roster_df = roster_df.append(player_df)
    # st.dataframe(roster_df)
    return roster_df

roster_team = st.text_input('Enter a team ID from the table above:', '22')
roster_year = st.text_input('Enter a season ID (eg. 20182019 for the 2018-2019 season)', '20182019')
st.write("Here is that team's roster for that year:")
st.dataframe(get_roster(team_id=int(roster_team), season_id=int(roster_year)))

st.write("Now we'll get a roster that covers all players that were active in a range of season. NOTE: this uses the team entered earlier.")

@st.cache
def merge_team_rosters(team_id=22, season_id_list=[20152016, 20162017, 20172018, 20182019]):
    merged_roster_df = pd.DataFrame()
    for season in season_id_list:
        roster_df = get_roster(team_id=team_id, season_id=season)
        merged_roster_df = merged_roster_df.append(roster_df)
    merged_roster_df.drop_duplicates(inplace=True)
    return merged_roster_df

merged_roster = merge_team_rosters(team_id=roster_team)
st.dataframe(merged_roster)

st.write("We'll start collecting time-series sets now for exploration.")
roster_player = st.text_input('Enter a player ID from the table above:', '8477934')

@st.cache
def get_player_season_game_stats(player_id=8478402, season_id=20182019):
    stats_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/people/{0}/stats?stats=gameLog&season={1}'.format(player_id, season_id))
    if stats_response.status_code != 200:
        st.error("Failed attempt to get player {0}'s game stats for season {1}.".format(player_id, season_id))
        return
    stats = stats_response.content
    stats_df = pd.read_json(stats)
    stats_df = json_normalize(stats_df.stats)
    stats_df = stats_df.splits
    stats_array = stats_df.array
    stats_list = stats_array[0]
    stats_df = pd.DataFrame()
    for game in stats_list:
        game_df = pd.DataFrame.from_dict(game).transpose()
        clean_df = pd.DataFrame()
        for stat_type, stat_series in game_df.iterrows():
            if stat_type != 'stat': # the 'stat' stat type contains non-unique but desired values
                try:
                    stat_series.drop_duplicates(inplace=True)
                except SystemError:
                    pass
            stat_series.dropna(inplace=True)
            stat_df = pd.DataFrame(stat_series).transpose()
            new_columns = [stat_type + column.capitalize() for column in stat_df.columns if len(stat_df.columns) != 1]
            if len(new_columns) == 0: new_columns = stat_df.index
            stat_df.reset_index(drop=True, inplace=True)
            stat_df.columns = new_columns
            clean_df = pd.concat([clean_df, stat_df], axis=1)
        game_df = clean_df.drop('gameContent', axis=1)
        game_df.set_index('gameGamepk', inplace=True)
        stats_df = stats_df.append(game_df)
    return stats_df

player_df = get_player_season_game_stats(player_id=roster_player)
player_df.sort_index(inplace=True)
st.write('{0} ({1}) stat dataframe for {2} season:'.format(merged_roster.loc[np.int(roster_player), 'name'], merged_roster.loc[np.int(roster_player), 'position'], roster_year))
st.dataframe(player_df)
st.write('Season (sorted) point history:')
st.dataframe(player_df.loc[:, 'statPoints'])

@st.cache
def augment_player_dataframe(player_df=player_df):
    augmented_df = player_df
    points_series = augmented_df.loc[:, 'statPoints']
    points_series = points_series.cumsum()
    points_series.name = 'cum' + points_series.name.capitalize()
    augmented_df = pd.concat([augmented_df, points_series], axis=1)
    st.dataframe(augmented_df)
    return augmented_df

augmented_df = augment_player_dataframe(player_df=player_df)
line_fig = px.line(augmented_df, x='date', y='cumStatpoints')
line_fig.update_layout(title_text='test', title_x=0.5)
line_fig.update_yaxes(title_text='Cumulative Points')
st.plotly_chart(line_fig)

@st.cache
def assemble_stat_series(player_id_list=[8477934, 8476365], season_id=20182019, stat='cumStatpoints'):
    for player_id in player_id_list:
        player_name = None # TODO: write call to get player list; should be replaced with a lookup from a static data source
        player_df = augment_player_dataframe(get_player_season_game_stats(player_id=player_id, season_id=season_id))
    return