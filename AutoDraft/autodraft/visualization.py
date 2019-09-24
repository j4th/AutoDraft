"""
Module to visualize model inputs, outputs, and performance
"""
from datetime import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool
from bokeh.plotting import figure

def calculate_predictions(data, results, player_name='Leon Draisaitl', target='cumStatpoints'):
    """ calculate predictions from residuals and the original data"""
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
    full_frame['predictions'] = full_frame \
                                    .apply(lambda row: row.cumStatpoints - row.residuals, axis=1)
    return full_frame

def return_intervals(results, player_name='Leon Draisaitl'):
    """ return the prediction intervals from a results dataframe"""
    lows = results.loc[player_name, 'intervalLow']
    highs = results.loc[player_name, 'intervalHigh']
    try:
        intervals = pd.DataFrame({'low':lows, 'high':highs})
    except ValueError:
        intervals = pd.DataFrame({'low':lows.tolist(), 'high':highs.tolist()})
    return intervals

def plot_actual_predictions_series(data,
                                   results,
                                   target='cumStatpoints',
                                   metric='Rmse',
                                   player_name='Leon Draisaitl'):
    """ plots the real and predicted time series along with confidence intervals for a player"""
    series_dataframe = calculate_predictions(data, results, player_name=player_name, target=target)
    intervals = return_intervals(results, player_name)
    dates = series_dataframe.index.values.astype(np.datetime64)
    start_date = dt.strptime('2018-10-03', '%Y-%m-%d')

    real_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe[target]))
    pred_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['predictions']))
    interval_dates = dates[-intervals.shape[0]:].reshape(-1, 1)
    interval_dates = np.hstack((interval_dates, interval_dates))

    player_line = figure(title=('{0}({1},{2},{3})({4},{5},{6},{7})'
                                '[Train RMSE: {8:.3f}, Test RMSE: {9:.3f}]') \
                                .format(player_name,
                                        results.loc[player_name, 'p'],
                                        results.loc[player_name, 'd'],
                                        results.loc[player_name, 'q'],
                                        results.loc[player_name, 'P'],
                                        results.loc[player_name, 'D'],
                                        results.loc[player_name, 'Q'],
                                        3, # TODO: undo hardcoding
                                        results.loc[player_name, 'train'+metric],
                                        results.loc[player_name, 'test'+metric],
                                        ), # TODO: change to MASE
                         plot_height=300,
                         plot_width=800,
                         tools="xpan",
                         toolbar_location='above',
                         x_axis_type="datetime",
                         x_axis_location="below",
                         x_range=(dates[0], dates[-1]),
                         background_fill_color="#efefef"
                         )

    hover_tool = HoverTool(tooltips=[("date", "@date"),
                                     ("points", "@points")
                                    ],
                           mode='vline'
                           )

    player_line.circle('date', 'points', source=real_source, line_color='blue', legend='actual')
    player_line.line('date', 'points', source=pred_source, line_color='red', legend='predicted')

    player_line.varea(x=interval_dates[:, 0],
                      y1=intervals.loc[:, 'high'],
                      y2=intervals.loc[:, 'low'],
                      fill_alpha=0.4,
                      color='red',
                      legend='predicted')

    player_line.legend.location = 'top_left'
    player_line.legend.click_policy = 'hide'
    player_line.add_tools(hover_tool)
    player_line.toolbar.active_multi = hover_tool
    # player_line.yaxis.axis_label('Cumulative Points')

    test_start = Span(location=start_date,
                      dimension='height', line_color='green',
                      line_dash='dashed', line_width=3)
    player_line.add_layout(test_start)

    select = figure(title=("Drag the middle and edges of the"
                           "selection box to change the range above"),
                    plot_height=130,
                    plot_width=800,
                    y_range=player_line.y_range,
                    x_axis_type="datetime",
                    y_axis_type=None,
                    tools="",
                    toolbar_location=None,
                    background_fill_color="#efefef",
                    x_range=(dates[0], dates[-1]))

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
# def get_hists(results_list, metric_list=None, result_names=None, edges_method='min'):
#     """ get the histograms and edges from a set of results """
#     if metric_list is None:
#         metric_list = ['testRmse']
#     if result_names is None:
#         result_names = ['Raw', 'Yeo-Johnson']
#     hists = []
#     edges_list = []
#     for result in results_list:
#         for metric in metric_list:
#             hist, edges = np.histogram(result[metric])
#             hists.append(hist)
#             edges_list.append(edges)
#     edge_min = edges_list[0][-1]
#     edge_max = edges_list[0][-1] # I cheated here
#     edge_loc = None
#     for i, edges in enumerate(edges_list):
#         edge_range = edges[-1] - edges[0]
#         if edges_method == 'max':
#             if edge_range > edge_max:
#                 edge_loc = i
#         else:
#             if edge_range < edge_min:
#                 edge_loc = i
#     if not edge_loc:
#         edge_loc = 0
#     edges = edges_list[edge_loc]
#     st.dataframe(edges)
#     return hists, edges

# def plot_hists(result_list, metric_list=None, result_names=None):
#     """plot a set of histograms from a list of results for the given metrics"""
#     if metric_list is None:
#         metric_list = ['testRmse']
#     if result_names is None:
#         result_names = ['Raw', 'Yeo-Johnson']
#     fig = figure(plot_height=600,
#                  plot_width=600,
#                  title="Histogram of ARIMA RMSE's",
#                  x_axis_label='RMSE',
#                  y_axis_label='# of players')

#     results_df = pd.DataFrame()
#     for result, name in zip(result_list, result_names):
#         metrics_df = pd.DataFrame()
#         for metric in metric_list:
#             metric_df = result[metric]
#             metrics_df = pd.concat([metrics_df, metric_df], axis=1)
#         metrics_df.columns = [column + name for column in metrics_df.columns]
#         results_df = pd.concat([results_df, metrics_df], axis=1)
#         # TODO: I don't think multiple metrics will play nice here...
#         # result_df = pd.concat([result[metric] for metric in metric_list], axis=1)
#     max_range = results_df.max() - results_df.min()
#     st.dataframe(max_range)

#     # TODO: add scaling of axes back in
#     hists, edges = get_hists(metric_list, result_list, result_names, edges_method='max')
#     # TODO: handle colors/length mismatch
#     for result_name, hist, color in zip(result_names, hists, ['blue', 'red', 'green']):
#         fig.quad(bottom=0,
#                  top=hist,
#                  left=edges[:-1],
#                  right=edges[1:],
#                  fill_color=color,
#                  line_color='black',
#                  legend=result_name)

#     fig.legend.location = 'top_right'
#     fig.legend.click_policy = 'hide'

#     st.bokeh_chart(fig)
#     return fig
