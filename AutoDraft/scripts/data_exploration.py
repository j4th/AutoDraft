"""
Script for downloading and exploring data available on the NHL API
"""
import streamlit as st
import numpy as np
import autodraft.api as api
# import plotly.express as px

def main():
    st.title('NHL API Dataset Assembly')
    st.write('This report will cover assembling a player-level'
             'game-by-game dataset for 4 NHL seasons.')
    st.write('For the most complete dataset, we will go through'
             'each game on each day so as to get all stats,'
             'but only for those players that played in the game.')

    st.write('Here is the full dataframe of NHL seasons with available stats:')
    seasons_df = api.get_seasons(streamlit=True)
    st.dataframe(seasons_df)

    st.write("Let's grab the 4 most recent, COMPLETE, seasons.")
    seasons_df = seasons_df.iloc[-5:-1, :]
    st.dataframe(seasons_df)
    st.write('Now on to getting our team rosters.')

    st.dataframe(api.get_teams(streamlit=True))
    st.write('From here, we will get the rosters for each team in the seasons we want.')

    # roster_team = st.text_input('Enter a team ID from the table above:', '22')
    roster_team = '22'
    # roster_year = st.text_input('Enter a season ID (eg. 20182019 for the 2018-2019 season)', '20182019')
    roster_year = '20182019'

    st.write("Here is that team's roster for that year:")
    st.dataframe(api.get_roster(team_id=int(roster_team), season_id=int(roster_year), streamlit=True))

    st.write("Now we'll get a roster that covers all players that were active"
             "in a range of season. NOTE: this uses the team entered earlier.")

    merged_roster = api.merge_team_rosters(team_id=roster_team)
    st.dataframe(merged_roster)

    st.write("We'll start collecting time-series sets now for exploration.")
    # roster_player = st.text_input('Enter a player ID from the table above:', '8477934')
    roster_player = '8477934'

    player_df = api.get_player_season_game_stats(player_id=roster_player, streamlit=True)
    player_df.sort_index(inplace=True)
    st.write('{0} ({1}) stat dataframe for {2} season:'.format(merged_roster.loc[np.int(roster_player), 'name'],
                                                               merged_roster.loc[np.int(roster_player), 'position'],
                                                               roster_year))
    st.dataframe(player_df)
    st.write('Season (sorted) point history:')
    st.dataframe(player_df.loc[:, 'statPoints'])

    combined_player_df = api.get_combined_player_season_game_stats(player_id=roster_player)
    combined_player_df.sort_index(inplace=True)

    augmented_df = api.augment_player_dataframe(player_df=combined_player_df)

    st.write('{0} ({1}) combined and augmented stat dataframe for {2} seasons:' \
             .format(merged_roster.loc[np.int(roster_player), 'name'],
                     merged_roster.loc[np.int(roster_player), 'position'],
                     3))
    st.dataframe(augmented_df)

    # line_fig = px.line(augmented_df, x='gameNumber', y='cumStatpoints')
    # line_fig.update_layout(title_text='test', title_x=0.5)
    # line_fig.update_yaxes(title_text='Cumulative Points')
    # st.plotly_chart(line_fig)

    # multi_season_df = api.augment_player_dataframe(combined_player_df)
    # line_fig = px.line(multi_season_df, x='gameNumber', y='cumStatpoints')
    # line_fig.update_layout(title_text='test', title_x=0.5)
    # line_fig.update_yaxes(title_text='Cumulative Points')
    # st.plotly_chart(line_fig)

    # multiplayer_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(merge_team_rosters(team_id=roster_team).index), stat_list=[])
    # st.dataframe(multiplayer_df)
    # line_fig = px.line(multiplayer_df, x='date', y='cumStatpoints', color='name')
    # line_fig.update_layout(title_text='test', title_x=0.5, xaxis_rangeslider_visible=True, showlegend=True, legend=dict(x=1, y=-2))
    # line_fig.update_yaxes(title_text='Cumulative Points')
    # st.plotly_chart(line_fig)

    all_rosters = api.get_all_rosters(streamlit=True)
    st.dataframe(all_rosters)
    st.write(all_rosters.shape)

    full_df = api.assemble_multiplayer_stat_dataframe(player_id_list=list(all_rosters.index), stat_list=[])
    st.write(full_df.shape)

    # full_df.to_csv('./data/full_dataset_4_seasons.csv')

main()
