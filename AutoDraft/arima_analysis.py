import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool
from bokeh.models.glyphs import Patch
from bokeh.plotting import figure, show
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import NHL_API as nhl

@st.cache
def load_csv(path='./data/full_dataset_4_seasons.csv'):
    data = pd.read_csv(path)
    return data

# @st.cache
def load_pickle(path='./data/arima_results_m3.p'):
    data = pd.read_pickle(path)
    data.loc[:,'name'] = data.index
    data.drop_duplicates('name', inplace=True)
    return data

data = load_csv()
results = load_pickle()
results_yj = load_pickle('./data/arima_results_m3_yj.p')

st.text('Stats data shape: {0}\nARIMA results shape: {1}'.format(data.shape, results.shape))
st.write('Stat dataframe head:')
st.dataframe(data.head())
st.write('ARIMA results dataframe:')
st.dataframe(results)
st.write('Yeo-Johnsoned ARIMA results dataframe:')
st.dataframe(results_yj)

def get_hists(metric_list=['testRmse'], results=[results, results_yj], result_names=['Raw', 'Yeo-Johnson'], range_result=None, range_metric=None, edges_method='min'):
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

def plot_hists(metric_list=['testRmse'], results=[results, results_yj], result_names=['Raw', 'Yeo-Johnson']):
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

# plot_hists(['testRmse'])

test_player = st.text_input('Player to predict:', 'Leon Draisaitl')

# @st.cache
def calculate_predictions(data=data, results=results, player_name=test_player, target='cumStatpoints'):
    test_results = results.loc[test_player, :]
    test_residuals = test_results.testResiduals
    train_residuals = test_results.trainResiduals
    test_real = data[data['name'] == test_player].loc[:, ['date', target]]
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
    full_frame['predictions'] = full_frame.apply(lambda row: row.cumStatpoints - row.residuals, axis=1)
    # st.dataframe(full_frame)
    return full_frame, player_name

def return_intervals(results=results, player_name=test_player):
    lows = results.loc[test_player, 'intervalLow']
    highs = results.loc[test_player, 'intervalHigh']
    # st.write(type(lows))
    # for data in [lows, highs]:
    #     if type(data) == type(np.ndarray((1,1))): data = data.tolist() 
    # st.write(type(lows))
    try:
        intervals = pd.DataFrame({'low':lows, 'high':highs})
    except ValueError:
        intervals = pd.DataFrame({'low':lows.tolist(), 'high':highs.tolist()})
    return intervals

# test_intervals = return_intervals()

# @st.cache
def plot_actual_predictions_series(results, target='cumStatpoints', metric='Rmse',
                                    series_dataframe=calculate_predictions(player_name=test_player, target='cumStatpoints')[0],
                                    player_name=calculate_predictions(player_name=test_player, target='cumStatpoints')[1]):
    intervals = return_intervals(results, player_name)
    dates = series_dataframe.index.values.astype(np.datetime64)
    start_date = dt.strptime('2018-10-03', '%Y-%m-%d')

    real_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe[target]))
    pred_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['predictions']))
    interval_dates = dates[-intervals.shape[0]:].reshape(-1,1)
    interval_dates = np.hstack((interval_dates, interval_dates))
    interval_source = ColumnDataSource(data=dict(date=interval_dates, points=intervals))

    player_line = figure(title='{0}({1},{2},{3})({4},{5},{6},{7}) [Train RMSE: {8:.3f}, Test RMSE: {9:.3f}]'.format(test_player, 
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

transform = st.checkbox('Use transformed (YJ) data?')
if not transform:
    nhl.plot_actual_predictions_series(data, results, player_name=test_player)
else:
    nhl.plot_actual_predictions_series(data, results_yj, player_name=test_player)