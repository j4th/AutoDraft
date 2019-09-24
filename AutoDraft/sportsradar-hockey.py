"""
Incomplete script for retrieving and exploring data from sportsradar using VSCode's IPython implementation
"""
#%%[markdown]
# Let's look at what sort of data we can pull for the Minor leagues

#%%
import json
import requests as rqst
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

#%%[markdown]
# Let's grab our API keys from where they are stored.

#%%
global_hockey_API_key = open('./keys/sportsradar-global-ice-hockey.txt', mode='r').readline().strip()
NHL_API_key = open('./keys/sportsradar-NHL.txt', mode='r').readline().strip()

#%% [markdown]
# Now let's see what all we have on the "global" scale.

#%%
global_tournament_list_response = rqst.get('https://api.sportradar.us/hockey-t1/ice/en/tournaments.json?api_key={}'.format(global_hockey_API_key))
print(global_tournament_list_response.status_code)

#%%[markdown]
# We want to make sure it responds with a status code 200 (success).
# Now, let's import it into a dataframe and see what is there.

#%%
global_tournament_df = pd.read_json(global_tournament_list_response.content)
global_tournament_df.head()

#%%[markdown]
# Right away we can see there's some nesting in here, so we'll work with that to get it usable.


#%%
global_tournament_df = json_normalize(global_tournament_df['tournaments'])

#%%[markdown]
# Now, we'll just grab the relevant leagues from the dataframe.


#%%
global_league_list = ['sr:tournament:14361',
                        'sr:tournament:1454',
                        'sr:tournament:225',
                        'sr:tournament:844',
                        'sr:tournament:128',
                        'sr:tournament:261',
                        'sr:tournament:268',
                        'sr:tournament:236',
                        'sr:tournament:134',
                        'sr:tournament:237',
                        'sr:tournament:3',
                        'sr:tournament:769']
global_tournament_df = global_tournament_df[global_tournament_df['id'].isin(global_league_list)]

#%% [markdown]
# We'll use the WHL to work through finding what we want.
# Let's look awt how many season we have access to.

#%%
WHL_seasons_URL = 'https://api.sportradar.us/hockey-t1/ice/en/tournaments/sr:tournament:14361/seasons.json?api_key={0}'.format(global_hockey_API_key)
WHL_seasons_response = rqst.get(WHL_seasons_URL)
# WHL_seasons_df

#%%
