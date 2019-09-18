import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

data = load_csv()
test_player = data[data['name'] == 'Leon Draisaitl']
st.dataframe(test_player)

test_player.plot(x='date', y='cumStatpoints')
fig = plt.gcf()
st.pyplot(fig)

small_test_player = test_player.loc[:, ['date', 'cumStatpoints']]
smaller_test_player = small_test_player.set_index('date', drop=True)
st.dataframe(smaller_test_player)

autocorrelation_plot(smaller_test_player)
fig = plt.gcf()
st.pyplot(fig)

model = ARIMA(smaller_test_player, order=(3,1,1))
model_fit = model.fit(disp=0)
st.text(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
fig = plt.gcf()
st.pyplot(fig)

residuals.plot(kind='kde')
fig = plt.gcf()
st.pyplot(fig)
st.write(residuals.describe())