"""
Module to visualize model inputs, outputs, and performance
"""
from datetime import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool, FixedTicker, PrintfTickFormatter
from  bokeh.models.ranges import Range1d
from bokeh.plotting import figure
from scipy.stats.kde import gaussian_kde
from scipy.stats import ttest_ind
import colorcet as cc

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
    try:
        full_frame['predictions'] = full_frame \
                                        .apply(lambda row: row.cumStatpoints - row.residuals, axis=1)
    except Exception as e:
        raise e
    full_frame.loc[:, 'name'] = player_name
    return full_frame

def calculate_errors(residuals, start_date='2018-10-03', end_date=None):
    """ Calculates errors based on residuals """
    if start_date:
        residuals = residuals.loc[residuals.loc[:, 'date'] >= start_date]
    if end_date:
        residuals = residuals.loc[residuals.loc[:, 'date'] < end_date]
    num_residuals = len(residuals)
    residuals = pd.DataFrame(residuals.loc[:, 'residuals'])
    mfe = (residuals.sum() / num_residuals).tolist()[0]
    mae = (residuals.abs().sum() / num_residuals).tolist()[0]
    rmse = ((residuals.pow(2).sum() / num_residuals).pow(0.5)).tolist()[0]
    residuals = residuals.values
    residuals = [value.item() for value in residuals]
    return mfe, mae, rmse

def calculate_residuals(predictions, start_date='2018-10-03', end_date=None):
    if start_date:
        predictions = predictions.loc[predictions.loc[:, 'date'] >= start_date]
    if end_date:
        predictions = predictions.loc[predictions.loc[:, 'date'] < end_date]
    residuals = predictions.loc[:, 'cumStatpoints'] - predictions.loc[:, 'predictions']
    residuals = pd.DataFrame({'date':predictions.loc[:, 'date'],
                              'name':predictions.loc[:, 'name'],
                              'residuals':residuals})
    return residuals

def calculate_error_df(predictions, drift_loc, start_date='2018-10-03', end_date=None, use_mase=True):
    if use_mase:
        drift_df = pd.read_csv(drift_loc)
    error_df = pd.DataFrame()
    for player in predictions.loc[:, 'name'].unique():
        mfe, mae, rmse = calculate_errors(
                            calculate_residuals(predictions.loc[predictions.loc[:, 'name'] == player], 
                                                start_date=start_date,
                                                end_date=end_date),
                            start_date=start_date,
                            end_date = end_date)

        if use_mase:

            player_drift = drift_df.loc[drift_df.loc[:, 'name'] == player]
            drift_residuals = calculate_residuals(player_drift, 
                                                    start_date='2018-10-03',
                                                    end_date=None)
            _, drift_mae, _ = calculate_errors(
                                drift_residuals,
                                start_date='2018-10-03',
                                end_date = None)
            if drift_mae != 0:
                mase = mae / drift_mae
            else:
                mase = float('NaN')
        if use_mase:
            player_df = pd.DataFrame({'name':player,
                                    'mase':mase,
                                    'mfe':mfe,
                                    'mae':mae,
                                    'rmse':rmse}, index=[player])
        else:
            player_df = pd.DataFrame({'name':player,
                                    'mfe':mfe,
                                    'mae':mae,
                                    'rmse':rmse}, index=[player])
        error_df = pd.concat([error_df, player_df])
        error_df = error_df.reset_index(drop=True)
    return error_df

def generate_error_df(predictions_file_list, data_loc, drift_loc, metric='mase'):
    output_df = pd.DataFrame()
    for prediction_file in predictions_file_list:
        if metric == 'mase' and 'drift' in prediction_file:
            continue
        if prediction_file.split('.')[-1] == 'p':
            data = pd.read_csv(data_loc)
            results = pd.read_pickle(prediction_file)
            results.loc[:, 'name'] = results.index
            predictions = pd.DataFrame()
            for player in  results.loc[:, 'name'].unique():
                try:
                    player_prediction = calculate_predictions(data, results, player_name=player)
                except AttributeError:
                    continue
                predictions = pd.concat([predictions, player_prediction])
        else:
            predictions = pd.read_csv(prediction_file)
        if 'drift' in prediction_file:
            error_df = calculate_error_df(predictions, drift_loc, start_date='2018-10-03', end_date=None)
        else:
            error_df = calculate_error_df(predictions, drift_loc)
        error_df = pd.DataFrame({'name.{}'.format(prediction_file.split('/')[-1].split('.')[0]): error_df.loc[:, 'name'],
                                 prediction_file.split('/')[-1].split('.')[0]: error_df.loc[:, metric]})
        output_df = pd.concat([output_df, error_df], axis=1)
    return output_df

def test_errors(error_df, dist1='arima_results_m3', dist2='deepar_truncated_results_default', report=False):
    t, p = ttest_ind(error_df.loc[:, dist1], error_df.loc[:, dist2], equal_var=False, nan_policy='omit')
    if report:
        st.text('{0} model mean MASE: {1:.3f} (std.: {2})\n{3} model mean MASE: {4:.3f} (std.: {5})' \
                 .format(dist1,
                         error_df.loc[:, dist1].mean(),
                         error_df.loc[:, dist1].std(),
                         dist2,
                         error_df.loc[:, dist2].mean(),
                         error_df.loc[:, dist2].std()))
    return t, p

def calculate_par(player, joe):
    foo = None
    # st.dataframe(player)
    # st.dataframe(joe)
    return

def assemble_diagnoses(data, errors):
    return

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
                                   model_errors,
                                   joe=None,
                                   par=False,
                                   model='arima',
                                   target='cumStatpoints',
                                   metric='mase',
                                   player_name='Leon Draisaitl',
                                   deepar_model_name='deepar_truncated_results_unit_s_ne300_lr1e-3_bs64_nl3_cl3.csv'):
    """ plots the real and predicted time series along with confidence intervals for a player"""
    if results is None:
        model = 'deepar'
    if model == 'arima':
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        player_errors = model_errors.loc[model_errors.loc[:, 'name.arima_results_m3'] == player_name]
        try:
            player_error = player_errors['arima_results_m3'].values.astype(float).item()
        except ValueError:
            st.write('Unfortunately, Auto-ARIMA was unable to find a suitable model for this player.')
            return
        series_dataframe = calculate_predictions(data, results, player_name=player_name, target=target)
        intervals = return_intervals(results, player_name)
        dates = series_dataframe.index.values.astype(np.datetime64)
        joe = joe.loc[joe.loc[:, 'date'].isin(player_df.date)]
        joe = joe.set_index('date')
        compare_dataframe = joe.loc[:, 'cumStatpoints']
        compare_dataframe = compare_dataframe.sort_index()
    elif model == 'deepar':
        player_df = data.loc[data.loc[:, 'name'] == player_name]
        if model_errors is not None:
            player_errors = model_errors.loc[model_errors.loc[:, 'name.{}'.format(deepar_model_name)] == player_name]
            try:
                player_error = player_errors[deepar_model_name].values.astype(float).item()
            except ValueError:
                st.write('This player was not able to be evaluated.')
        else:
            player_df = player_df.sort_values('gameNumber', ascending=True)
            player_df = player_df.set_index('gameNumber')
            # player_df.dropna(axis=1, subset=['predictions'])
        # _, _, rmse = calculate_errors(calculate_residuals(player_df))
        if model_errors is not None:
            series_dataframe = player_df.loc[:, [target, 'predictions']]
        else:
            series_dataframe = player_df.loc[:, 'predictions']
        series_dataframe = series_dataframe.sort_index()
        intervals = player_df.loc[:, ['high', 'low']].dropna()
        intervals = intervals.sort_index()
        if model_errors is not None:
            dates = series_dataframe.index.values.astype(np.datetime64)
        else:
            gameNumbers = series_dataframe.index.values.astype(int)
        # st.write(dates)
        if model_errors is not None:
            joe = joe.loc[joe.loc[:, 'date'].isin(series_dataframe.index)]
            joe = joe.set_index('date')
            compare_dataframe = joe.loc[:, 'cumStatpoints']
            compare_dataframe = compare_dataframe.sort_index()
        if par:
            calculate_par(series_dataframe, compare_dataframe)
    elif data is None and results is None:
        player_name = 'Joe Schmo'
        joe = joe.set_index('date')
        series_dataframe = joe.loc[:, 'cumStatpoints']
        series_dataframe = series_dataframe.sort_index()
        dates = series_dataframe.index.values.astype(np.datetime64)
    start_date = dt.strptime('2018-10-03', '%Y-%m-%d')

    if model_errors is not None:
        # st.dataframe(model_errors)
        real_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe[target]))
        compare_source = ColumnDataSource(data=dict(date=dates, points=compare_dataframe))
        pred_source = ColumnDataSource(data=dict(date=dates, points=series_dataframe['predictions']))
        interval_dates = dates[-intervals.shape[0]:].reshape(-1, 1)
        interval_dates = np.hstack((interval_dates, interval_dates))
    else:
        pred_source = ColumnDataSource(data=dict(gameNumber=gameNumbers, points=series_dataframe))
        interval_games = gameNumbers[-intervals.shape[0]:].reshape(-1, 1)
        interval_games = np.hstack((interval_games, interval_games))

    if model_errors is not None:
        y_max = max([series_dataframe.loc[:, target].max(), joe.loc[:, target].max()]) + 50
    else:
        try:
            y_min = series_dataframe.values[0]
        except IndexError:
            st.write('This player was not able to be predicted.')
            return
        y_max = series_dataframe.max() + 50

    if model == 'arima':
        player_line = figure(title=('{0}({1},{2},{3})({4},{5},{6},{7}) ({8}: {9:.3f})') \
                                    .format(player_name,
                                            results.loc[player_name, 'p'],
                                            results.loc[player_name, 'd'],
                                            results.loc[player_name, 'q'],
                                            results.loc[player_name, 'P'],
                                            results.loc[player_name, 'D'],
                                            results.loc[player_name, 'Q'],
                                            3, # TODO: undo hardcoding
                                            metric.upper(),
                                            player_error), # TODO: change to MASE
                            plot_height=300,
                            plot_width=800,
                            tools="xpan",
                            toolbar_location='above',
                            x_axis_type="datetime",
                            x_axis_location="below",
                            x_range=(start_date, dates[-1]),
                            y_range=(0, y_max),
                            x_axis_label='Game Date',
                            y_axis_label='Cumulative Points',
                            background_fill_color="#efefef"
                            )
    else:
        if model_errors is not None:
            player_line = figure(title=('{0} ({1}: {2:.3f})'.format(player_name, metric.upper(), player_error)), # TODO: add error
                                plot_height=300,
                                plot_width=800,
                                tools="xpan",
                                toolbar_location='above',
                                x_axis_type="datetime",
                                x_axis_location="below",
                                x_range=(start_date, dates[-1]),
                                y_range=(0, y_max),
                                x_axis_label='Game Date',
                                y_axis_label='Cumulative Points',
                                background_fill_color="#efefef"
                                )
        else:
            player_line = figure(title=('{0}'.format(player_name)), # TODO: add error
                                plot_height=300,
                                plot_width=800,
                                tools="xpan",
                                toolbar_location='above',
                                x_axis_location="below",
                                x_range=(0, 82),
                                y_range=(y_min, y_max),
                                x_axis_label='Game Date',
                                y_axis_label='Cumulative Points',
                                background_fill_color="#efefef"
                                )
    # elif data is None and results is None:
    #     player_line = figure(title=('{0}'.format(player_name)), # TODO: add error
    #                          plot_height=300,
    #                          plot_width=800,
    #                          tools="xpan",
    #                          toolbar_location='above',
    #                          x_axis_type="datetime",
    #                          x_axis_location="below",
    #                          x_range=(dates[0], dates[-1]),
    #                          background_fill_color="#efefef"
    #                          )

    # hover_tool = HoverTool(tooltips=[("date", "@date"),
    #                                  ("points", "@points")
    #                                 ],
    #                        mode='vline'
    #                        )

    if model_errors is not None:
        player_line.line('date', 'points', source=real_source, line_color='blue', legend='actual')
        player_line.line('date', 'points', source=compare_source, line_color='green', legend='average')
        player_line.circle('date', 'points', source=pred_source, line_color='red', fill_color='red', legend='predicted')
        player_line.varea(x=interval_dates[:, 0],
                y1=intervals.loc[:, 'high'],
                y2=intervals.loc[:, 'low'],
                fill_alpha=0.4,
                color='red',
                legend='predicted')
    else:
        player_line.circle('gameNumber', 'points', source=pred_source, line_color='red', fill_color='red', legend='predicted')
        player_line.varea(x=interval_games[:, 0],
                y1=intervals.loc[:, 'high'],
                y2=intervals.loc[:, 'low'],
                fill_alpha=0.4,
                color='red',
                legend='predicted')



    player_line.legend.location = 'bottom_right'
    player_line.legend.click_policy = 'hide'
    # player_line.add_tools(hover_tool)
    # player_line.toolbar.active_multi = hover_tool
    # player_line.yaxis.axis_label('Cumulative Points')

    test_start = Span(location=start_date,
                      dimension='height', line_color='green',
                      line_dash='dashed', line_width=3)
    player_line.add_layout(test_start)

    if model_errors is not None:
        select = figure(title=("Drag the middle and edges of the"
                            "selection box to change the range above"),
                        plot_height=130,
                        plot_width=800,
                        y_range=player_line.y_range,
                        x_axis_type="datetime",
                        y_axis_type=None,
                        x_axis_label='Game Date',
                        y_axis_label='Cumulative Points',
                        tools="",
                        toolbar_location=None,
                        background_fill_color="#efefef",
                        x_range=(dates[0], dates[-1]))
    else:
        select = figure(title=("Drag the middle and edges of the"
                        "selection box to change the range above"),
                        plot_height=130,
                        plot_width=800,
                        y_range=player_line.y_range,
                        x_axis_type="datetime",
                        y_axis_type=None,
                        x_axis_label='Game Date',
                        y_axis_label='Cumulative Points',
                        tools="",
                        toolbar_location=None,
                        background_fill_color="#efefef",
                        x_range=(0, 82))

    # range_tool = RangeTool(x_range=Range1d(start_date, dates[-1]))
    range_tool = RangeTool(x_range=player_line.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    if model_errors is not None:
        select.line('date', 'points', source=real_source, line_color='blue')
        select.line('date', 'points', source=compare_source, line_color='green')
        select.circle('date', 'points', source=pred_source, line_color='red', fill_color='red')
    else:
        select.circle('gameNumber', 'points', source=pred_source, line_color='red', fill_color='red')

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    select.add_layout(test_start)

    chart = column(player_line, select)

    st.bokeh_chart(chart)
    # st.write("{}'s dataframe:".format(player_name))
    # st.dataframe(series_dataframe)
    return chart

def ridge(category, data, scale=3):
    return list(zip([category]*len(data), scale*data))

def ridge_plots(model_df):
    models = list(model_df.keys())
    models.reverse()
    for i, model in enumerate(models):
        if 'name' in model:
            models.pop(i)

    palette = [cc.rainbow[i*15] for i in range(len(models))]

    x = np.linspace(-20, 110, 500)

    source = ColumnDataSource(data=dict(x=x))

    p = figure(y_range=models, plot_width=900, x_range=(0, 5), toolbar_location=None)

    for i, model in enumerate(models):
        if 'name' in model:
            continue
        pdf = gaussian_kde(model_df[model].dropna())
        y = ridge(model, pdf(x))
        source.add(y, model)
        p.patch('x', model, color=palette[i], alpha=0.6, line_color="black", source=source)

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.xaxis.ticker = FixedTicker(ticks=list(range(int(model_df.min().min()), 5, 1)))
    # p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    p.xgrid.ticker = p.xaxis[0].ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 2

    base = Span(location=1,
                dimension='height', line_color='black',
                line_dash='dashed', line_width=3)
    p.add_layout(base)

    st.bokeh_chart(p)
    return

## HERE BE DRAGONS
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
