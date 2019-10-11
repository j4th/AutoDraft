"""
This script is for finding the optimal distribution to be used in GluonTS
"""
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels as sm
import matplotlib.pyplot as plt
import autodraft.gluonts as glu

@st.cache
def get_data(path='../../data/input/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [stats.laplace,
                     stats.norm,
                     stats.nbinom,
                     stats.t,
                     stats.uniform
                    ]

    # Best holders
    best_distribution = stats.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def main():
    data = get_data()
    _, _, _, _, _, targets = glu.prep_df(data, column_list=['name', 'gameNumber', 'cumStatpoints'], streamlit=True, scale=True, target_output_df=True)
    test_gn = targets.loc[targets.loc[:, 'gameNumber'] == 1]
    st.dataframe(test_gn.head())
    best_name, best_params = best_fit_distribution(test_gn.loc[:, 'cumStatpoints'])
    st.write(best_name)
    st.write(best_params)

    dists_output = pd.DataFrame()
    for game in data.loc[:, 'gameNumber'].unique():
        gn_df = data.loc[data.loc[:, 'gameNumber'] == game]
        best_name, best_params = best_fit_distribution(gn_df.loc[:, 'cumStatpoints'])
        results_df = pd.DataFrame({'gameNumber':game, 'best':best_name, 'params':best_params})
        dists_output = pd.concat([dists_output, results_df])
    st.dataframe(dists_output)
    hist = dists_output.loc[:, 'best'].hist()
    fig = plt.gcf()
    st.pyplot(fig)



main()
