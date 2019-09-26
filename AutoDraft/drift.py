"""
This incomplete function will calculate a drift model for players as a baseline
"""
import pandas as pd
import streamlit as st

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data
