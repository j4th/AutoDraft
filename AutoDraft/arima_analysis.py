import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, RangeTool, Span
from bokeh.plotting import figure, show
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
# from pyramid.arima import auto_arima
import NHL_API as nhl

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
def load_pickle(path='./data/arima_results.p'):
    data = pd.read_pickle(path)
    return data

data = load_csv()
results = load_pickle()

st.text('Stats data shape: {0}\nARIMA results shape: {1}'.format(data.shape, results.shape))
st.write('Stat dataframe head:')
st.dataframe(data.head())
st.write('ARIMA results dataframe:')
st.dataframe(results)

def get_hists(metric_list, range_metric, results=results):
    hists = []
    for metric in metric_list:
        if metric == range_metric:
            hist, edges = np.histogram(results[metric])
            hists.append(hist)
        else:
            hist, _ = np.histogram(results[metric])
            hists.append(hist)
    return hists, edges

def plot_hists(metric_list, results=results):
    fig = figure(plot_height = 600, plot_width = 600, 
                    title = "Histogram of ARIMA RMSE's",
                    x_axis_label = 'RMSE', 
                    y_axis_label = '# of players')

    metric_df = pd.concat([results[metric] for metric in metric_list], axis=1)
    max_range = metric_df.max() - metric_df.min()
    max_range_metric = max_range.idxmax()


    hists, edges = get_hists(metric_list, range_metric=max_range_metric)
    # TODO: handle colors/length mismatch
    for metric, hist, color in zip(metric_list, hists, ['blue', 'red']):
        fig.quad(bottom=0, top=hist, 
                        left=edges[:-1], right=edges[1:], 
                        fill_color=color, line_color='black', legend=metric)

    fig.legend.location = 'top_right'
    fig.legend.click_policy = 'hide'

    st.bokeh_chart(fig)
    return fig

plot_hists(['testRmse', 'trainRmse'])

test_player = st.text_input('Player to predict:', 'Nico Hischier')

# @st.cache
def calculate_predictions(data=data, results=results, player_name=test_player):
    test_results = results.loc[test_player, :]
    test_residuals = test_results.testResiduals
    train_residuals = test_results.trainResiduals
    test_real = data[data['name'] == test_player].loc[:, ['date', 'cumStatpoints']]
    # test_real.set_index(pd.to_datetime(test_real['date']), inplace=True)

    # st.dataframe(test_real)
    # st.write([frame.shape for frame in [test_results, train_residuals, test_residuals]])

    full_residuals = pd.concat([train_residuals, test_residuals], axis=0)
    full_residuals.reset_index(inplace=True, drop=True)
    full_residuals.columns = ['residuals']
    test_real.reset_index(inplace=True, drop=True)

    # st.write(full_residuals.shape)
    # st.write(test_real.shape)
    full_frame = pd.concat([test_real, full_residuals], axis=1)
    full_frame['date'] = pd.to_datetime(full_frame['date'])
    full_frame.drop_duplicates(subset='date', keep='first', inplace=True)
    full_frame.set_index('date', drop=False, inplace=True)

    full_frame['predictions'] = full_frame.apply(lambda row: row.cumStatpoints + row.residuals, axis=1)
    # st.dataframe(full_frame)
    return full_frame, player_name

# @st.cache
def plot_actual_predictions_series(series_dataframe=calculate_predictions(player_name=test_player)[0],
                            player_name=calculate_predictions(player_name=test_player)[1]):
    dates = series_dataframe.index.values.astype(np.datetime64)
    start_date = dt.strptime('2018-10-03', '%Y-%m-%d')

    real_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['cumStatpoints']))
    pred_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['predictions']))

    player_line = figure(title='{0} (Train RMSE: {1:.3f}, Test RMSE: {2:.3f}'.format(test_player, results.loc[player_name, 'trainRmse'], results.loc[player_name, 'testRmse']),
                            plot_height=300, plot_width=800, tools="xpan", toolbar_location=None,
                            x_axis_type="datetime", x_axis_location="below", x_range=(dates[0], dates[-1]),
                            background_fill_color="#efefef")

    player_line.line('date', 'points', source=real_source, line_color='blue', legend='actual')
    player_line.line('date', 'points', source=pred_source, line_color='red', legend='predicted')
    player_line.legend.location = 'top_left'
    player_line.legend.click_policy = 'hide'
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

plot_actual_predictions_series(player_name=test_player)