import json
import streamlit as st
import numpy as np 
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime as dt
import requests as rqsts  
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool
from bokeh.models.glyphs import Patch
from bokeh.plotting import figure, show

def get_seasons(): # returns all seasons on record
    seasons_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/seasons')
    if seasons_response.status_code != 200:
        st.error('Failed attempt to get list of seasons.')
        return
    seasons = seasons_response.content
    seasons_df = pd.read_json(seasons)
    seasons_df = json_normalize(seasons_df.seasons)
    seasons_df.set_index('seasonId', inplace=True)
    return seasons_df

def get_teams(): # returns all teams FOR THE CURRENT SEASON
    teams_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/teams')
    if teams_response.status_code != 200:
        st.error('Failed attempt to get list of teams.')
        return
    teams = teams_response.content
    teams_df = pd.read_json(teams)
    teams_df = json_normalize(teams_df.teams)
    return teams_df

def get_roster(team_id=22, season_id=20182019): # returns roster for given team and season
    roster_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/teams/{0}?expand=team.roster&season={1}'.format(team_id, season_id))
    if roster_response.status_code != 200:
        st.error("Failed attempt to get team {0}'s roster for season {1}.".format(team_id, season_id))
        return
    roster = roster_response.content
    roster_df = pd.read_json(roster)
    roster_df = json_normalize(roster_df.teams)
    roster_list = roster_df['roster.roster'][0]
    roster_df = pd.DataFrame() # generate df to be filled
    for _, person in enumerate(roster_list): # populate the df with desired info for each player
        person_info = person['person']
        person_position = person['position']
        player_dict = {'name': person_info['fullName'],
                        'position': person_position['code']}
        player_df = pd.DataFrame(player_dict, index=[person_info['id']]) ## TODO: should change to not extract as int64
        roster_df = roster_df.append(player_df)
    return roster_df

def merge_team_rosters(team_id=22, season_id_list=[20152016, 20162017, 20172018, 20182019]): 
    # returns a roster that includes all players that played for a team across the seasons provided
    merged_roster_df = pd.DataFrame()
    for season in season_id_list:
        roster_df = get_roster(team_id=team_id, season_id=season)
        if type(roster_df) == None: continue # TODO: report this to the user
        merged_roster_df = merged_roster_df.append(roster_df)
    merged_roster_df.drop_duplicates(inplace=True)
    return merged_roster_df

def get_player_season_game_stats(player_id=8477934, season_id=20182019):
    # gets the game-by-game stats for a given player and season
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
    stats_df = pd.DataFrame() # generate stat df to be filled
    for game in stats_list:
        game_df = pd.DataFrame.from_dict(game).transpose() # imports and transposes df from returned game dict (json)
        clean_df = pd.DataFrame() # generate game df to be filled
        for stat_type, stat_series in game_df.iterrows():
            if stat_type != 'stat': # the 'stat' stat type contains non-unique but desired values
                try:
                    stat_series.drop_duplicates(inplace=True)
                except SystemError:
                    pass
            stat_series.dropna(inplace=True) # clean out NaN
            stat_df = pd.DataFrame(stat_series).transpose() # transpose the series so it fits properly into our df
            new_columns = [stat_type + column.capitalize() for column in stat_df.columns if len(stat_df.columns) != 1] # rename columns to prevent collision
            if len(new_columns) == 0: new_columns = stat_df.index # use the index if there are no new columns
            stat_df.reset_index(drop=True, inplace=True)
            stat_df.columns = new_columns # rename columns
            clean_df = pd.concat([clean_df, stat_df], axis=1) # add the game to the df
        game_df = clean_df.drop('gameContent', axis=1) # replace the dirty df with our clean one
        game_df.set_index('gameGamepk', inplace=True) # set the indices to be the unique game identifier
        stats_df = stats_df.append(game_df)
    return stats_df

def get_combined_player_season_game_stats(player_id=8477934, season_id_list=[20162017, 20172018, 20182019]): # TODO: add function to set whether each season should be individual cumulative totals or cumulative across all seasons
    # returns player game-by-game stats across multiple seasons
    full_df = pd.DataFrame()
    for season_id in season_id_list:
        season_df = get_player_season_game_stats(player_id=player_id, season_id=season_id)
        full_df = pd.concat([full_df, season_df])
    full_df.drop_duplicates(subset='date', keep='first', inplace=True) # drop all duplicate entries (resulting from playing for multiple teams)
    return full_df

def augment_player_dataframe(player_df, cumulative_stat_list=['statPoints']): # generates cumulative totals of stats
    augmented_df = player_df
    augmented_df.sort_index(inplace=True) # make sure everything is in order
    for stat in cumulative_stat_list:
        try:
            stat_series = augmented_df.loc[:, stat] # grab a stat
        except KeyError: # TODO: verify why there are no points for these players; THINK its because I asked for seasons that didn't exist. still necessary?
            stat_series = pd.DataFrame({'cum'+stat.capitalize(): [None for _ in range(len(augmented_df))]})
        stat_series = stat_series.cumsum() 
        try:
            stat_series.name = 'cum' + stat_series.name.capitalize() # rename the stat column
        except AttributeError:
            pass
        augmented_df = pd.concat([augmented_df, stat_series], axis=1)
    augmented_df.insert(0, 'gameNumber', [i+1 for i in range(len(augmented_df))]) # add game numbers column
    return augmented_df

def get_player_name_position(player_id=8477934): # returns CURRENT basic player name and position
    player_response = rqsts.get('https://statsapi.web.nhl.com/api/v1/people/{0}/'.format(player_id))
    player = player_response.content
    player = pd.read_json(player)
    player = player.people[0]
    player_name = player['fullName']
    player_position = player['primaryPosition']['code']
    return player_name, player_position

def assemble_multiplayer_stat_dataframe(player_id_list=[8477934, 8476356, 8473468], season_id_list=[20152016, 20162017, 20172018, 20182019], stat_list=['cumStatpoints'], shape='cols'):
    # returns game-by-game stats for given list of players across given seasons (can specify sspecific stats for smaller returns)
    multiplayer_df = pd.DataFrame() # TODO: add a progress bar
    for player_id in player_id_list: 
        player_name, player_position = get_player_name_position(player_id)
        if player_position == 'G': continue # don't include goalies
        if len(season_id_list) == 1: # handle single or multiple season input lists
            player_df = augment_player_dataframe(get_player_season_game_stats(player_id=player_id, season_id=season_id_list[0]))
        else:
            player_df = augment_player_dataframe(get_combined_player_season_game_stats(player_id=player_id, season_id_list=season_id_list))
        if len(stat_list) != 0: # if a specific stat is given, only grab that
            player_small_df = player_df.loc[:, ['date', 'gameNumber'] + stat_list] # keep the useful indices
        else:
            player_small_df = player_df # grab it all otherwise
        player_small_df.reset_index(drop=True, inplace=True) # get rid of messy index
        try:
            player_small_df.insert(0, 'name', [player_name for _ in range(len(player_small_df))])
        except ValueError: # still don't know why this is thrown sometimes
            st.dataframe(player_small_df)
            player_small_df.insert(0, 'errorName', [player_name for _ in range(len(player_small_df))])
        # player_small_df.set_index('gameNumber', inplace=True)
        # player_small_df.rename(columns={stat: player_name}, inplace=True)
        multiplayer_df = pd.concat([multiplayer_df, player_small_df], axis=0)
        multiplayer_df.to_csv('./data/assemble_multiplayer_stat_dataframe_TEMP.csv') # export temporary state
    multiplayer_df.reset_index(drop=True, inplace=True)
    if shape == 'rows': # TODO: fix this when required to feed data in a row-per-player fashion
        multiplayer_df = multiplayer_df.transpose()
        # multiplayer_df.set_index(player_id_list, inplace=True)
        try:
            multiplayer_df.insert(0, 'name', multiplayer_df.index)
        except ValueError:
            multiplayer_df.insert(0, 'errorName', multiplayer_df.index)
        multiplayer_df.insert(0, 'playerId', player_id_list)
        multiplayer_df.set_index('playerId', inplace=True)
    return multiplayer_df

def get_all_rosters(season_id_list=[20152016, 20162017, 20172018, 20182019]): # returns roster of all players across a given list of seasons
    teams_df = get_teams()
    team_ids = teams_df.loc[:, 'id']
    full_roster_df = pd.DataFrame()
    for team_id in team_ids:
        team_full_roster = merge_team_rosters(team_id=team_id, season_id_list=season_id_list) # TODO: save player id's
        full_roster_df = pd.concat([full_roster_df, team_full_roster])
    return full_roster_df

def calculate_predictions(data, results, player_name='Leon Draisaitl', target='cumStatpoints'):
    test_results = results.loc[player_name, :]
    test_residuals = test_results.testResiduals
    train_residuals = test_results.trainResiduals
    test_real = data[data['name'] == player_name].loc[:, ['date', target]]
    full_residuals = pd.concat([train_residuals, test_residuals], axis=0)
    full_residuals.reset_index(inplace=True, drop=True)
    full_residuals.columns = ['residuals']
    test_real.reset_index(inplace=True, drop=True)
    full_frame = pd.concat([test_real, full_residuals], axis=1)
    full_frame['date'] = pd.to_datetime(full_frame['date'])
    full_frame.drop_duplicates(subset='date', keep='first', inplace=True)
    full_frame.set_index('date', drop=False, inplace=True)
    full_frame['predictions'] = full_frame.apply(lambda row: row.cumStatpoints - row.residuals, axis=1)
    return full_frame

def return_intervals(results, player_name='Leon Draisaitl'):
    lows = results.loc[player_name, 'intervalLow']
    highs = results.loc[player_name, 'intervalHigh']
    try:
        intervals = pd.DataFrame({'low':lows, 'high':highs})
    except ValueError:
        intervals = pd.DataFrame({'low':lows.tolist(), 'high':highs.tolist()})
    return intervals

def plot_actual_predictions_series(data, results, target='cumStatpoints', metric='Rmse', player_name='Leon Draisaitl'):
    series_dataframe = calculate_predictions(data, results, player_name=player_name, target=target)
    intervals = return_intervals(results, player_name)
    dates = series_dataframe.index.values.astype(np.datetime64)
    start_date = dt.strptime('2018-10-03', '%Y-%m-%d')

    real_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe[target]))
    pred_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['predictions']))
    interval_dates = dates[-intervals.shape[0]:].reshape(-1,1)
    interval_dates = np.hstack((interval_dates, interval_dates))
    interval_source = ColumnDataSource(data=dict(date=interval_dates, points=intervals))

    player_line = figure(title='{0}({1},{2},{3})({4},{5},{6},{7}) [Train RMSE: {8:.3f}, Test RMSE: {9:.3f}]'.format(player_name, 
                                                                                                                    results.loc[player_name, 'p'],
                                                                                                                    results.loc[player_name, 'd'],
                                                                                                                    results.loc[player_name, 'q'],
                                                                                                                    results.loc[player_name, 'P'],
                                                                                                                    results.loc[player_name, 'D'],
                                                                                                                    results.loc[player_name, 'Q'],
                                                                                                                    3, # TODO: undo hardcoding once ARIMA results are regenerated
                                                                                                                    results.loc[player_name, 'train'+metric],
                                                                                                                    results.loc[player_name, 'test'+metric]), # TODO: change to MASE
                            plot_height=300, plot_width=800, tools="xpan", toolbar_location='above',
                            x_axis_type="datetime", x_axis_location="below", x_range=(dates[0], dates[-1]),
                            background_fill_color="#efefef")

    hover_tool = HoverTool(tooltips=[("date", "@date"),
                                    ("points", "@points")],
                            mode='vline')

    player_line.circle('date', 'points', source=real_source, line_color='blue', legend='actual')
    player_line.line('date', 'points', source=pred_source, line_color='red', legend='predicted')
    # player_line.patch(x=dates[-intervals.shape[0]:], y=intervals, line_color='red', alpha=0.4)

    player_line.varea(x=interval_dates[:, 0], y1=intervals.loc[:, 'high'], y2=intervals.loc[:, 'low'], fill_alpha=0.4, color='red', legend='predicted')
    # interval_glyph = Patch(x='date', y='points', fill_color="red", fill_alpha=0.4)
    # player_line.add_glyph(interval_source, interval_glyph)

    player_line.legend.location = 'top_left'
    player_line.legend.click_policy = 'hide'
    player_line.add_tools(hover_tool)
    player_line.toolbar.active_multi = hover_tool
    # player_line.yaxis.axis_label('Cumulative Points')

    test_start = Span(location=start_date,
                              dimension='height', line_color='green',
                              line_dash='dashed', line_width=3)
    player_line.add_layout(test_start)

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=130, plot_width=800, y_range=player_line.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef", x_range=(dates[0], dates[-1]))

    range_tool = RangeTool(x_range=player_line.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line('date', 'points', source=real_source, line_color='blue')
    select.line('date', 'points', source=pred_source, line_color='red')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    select.add_layout(test_start)

    chart = column(player_line, select)

    st.bokeh_chart(chart)
    st.write("{}'s dataframe:".format(player_name))
    st.dataframe(series_dataframe)
    return chart

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

def get_hists(results_list, metric_list=['testRmse'], result_names=['Raw', 'Yeo-Johnson'], edges_method='min'):
    hists = []
    edges_list = []
    for result, name in zip(results, result_names):
        for metric in metric_list:
            hist, edges = np.histogram(result[metric])
            hists.append(hist)
            edges_list.append(edges)
    edge_min = edges_list[0][-1]
    edge_max = edges_list[0][-1] # I cheated here
    edge_loc = None
    for i, edges in enumerate(edges_list):
        edge_range = edges[-1] - edges [0]
        if edges_method == 'max':
            if edge_range > edge_max: edge_loc = i
        else:
            if edge_range < edge_min: edge_loc = i
    if not edge_loc: edge_loc = 0
    edges = edges_list[edge_loc]
    st.dataframe(edges)
    return hists, edges

def plot_hists(result_list, metric_list=['testRmse'], result_names=['Raw', 'Yeo-Johnson']):
    fig = figure(plot_height = 600, plot_width = 600, 
                    title = "Histogram of ARIMA RMSE's",
                    x_axis_label = 'RMSE', 
                    y_axis_label = '# of players')

    results_df = pd.DataFrame()
    for result, name in zip(results, result_names):
        metrics_df = pd.DataFrame()
        for metric in metric_list:
            metric_df = result[metric]
            metrics_df = pd.concat([metrics_df, metric_df], axis=1)
        metrics_df.columns = [column + name for column in metrics_df.columns]
        results_df = pd.concat([results_df, metrics_df], axis=1)
        # result_df = pd.concat([result[metric] for metric in metric_list], axis=1) # TODO: I don't think multiple metrics will play nice here...
    max_range = results_df.max() - results_df.min()
    max_range_results = max_range.idxmax()
    st.dataframe(max_range)

    hists, edges = get_hists(metric_list, results, result_names, edges_method='max') # TODO: add scaling of axes back in
    # TODO: handle colors/length mismatch
    for result_name, hist, color in zip(result_names, hists, ['blue', 'red', 'green']):
        fig.quad(bottom=0, top=hist, 
                        left=edges[:-1], right=edges[1:], 
                        fill_color=color, line_color='black', legend=result_name)

    fig.legend.location = 'top_right'
    fig.legend.click_policy = 'hide'

    st.bokeh_chart(fig)
    return fig