from typing import Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from plotly.subplots import make_subplots
    from plotly import graph_objects as go
except:
    pass


def add_timeseries(fig, ts, color="green", name="time series", showlegend=False):
    timestamps = ts.index

    trace = go.Scatter(
        name=name,
        x=timestamps,
        y=ts[ts.columns[0]],
        mode="lines",
        line=dict(color=color),
        showlegend=showlegend,
    )

    fig.add_trace(trace)


def plot_timeseries(
    ts,
    color: Union[str, dict] = "green",
    fig=None,
    name="time series",
):

    showlegend = True
    if type(ts) == dict:
        data = ts
        if type(color) == str:
            color = {k: color for k in data}
    elif type(ts) == list:
        data = {}
        for k, ts_data in enumerate(ts):
            data[k] = ts_data
        if type(color) == str:
            color = {k: color for k in data}
    else:
        data = {}
        data["default"] = ts
        color = {"default": color}

    if fig is None:
        fig = go.Figure()

    first = True
    for key, ts in data.items():
        if not first:
            showlegend = False

        add_timeseries(fig, ts, color=color[key], showlegend=showlegend, name=name)
        first = False

    return fig


def plot_tsice_explanation(explanation, forecast_horizon):
    original_ts = pd.DataFrame(explanation["data_x"])
    perturbations = explanation["perturbations"]
    forecasts_on_perturbations = explanation["forecasts_on_perturbations"]

    new_perturbations = []
    new_timestamps = []
    pred_ts = []

    original_ts.index.freq = pd.infer_freq(original_ts.index)
    for i in range(1, forecast_horizon + 1):
        new_timestamps.append(original_ts.index[-1] + (i * original_ts.index.freq))

    for perturbation in perturbations:
        new_perturbations.append(pd.DataFrame(perturbation))

    for forecast in forecasts_on_perturbations:
        pred_ts.append(pd.DataFrame(forecast, index=new_timestamps))

    pred_original_ts = pd.DataFrame(
        explanation["current_forecast"], index=new_timestamps
    )

    # plot perturbed timeseries
    fig = plot_timeseries(
        ts=new_perturbations, color="lightgreen", name="perturbed timeseries samples"
    )
    # plot original timeseries
    plot_timeseries(
        ts=original_ts, fig=fig, name="input/original timeseries", color="green"
    )

    # plot varying forecast range
    plot_timeseries(
        ts=pred_ts, color="lightblue", fig=fig, name="forecast on perturbed samples"
    )

    # plot original forecast
    fig = plot_timeseries(
        ts=pred_original_ts, fig=fig, color="blue", name="original forecast"
    )

    fig.update_layout(template="plotly_white")
    fig.update_layout(
        title_text="Time Series Individual Conditional Expectation (TSICE) Plot"
    )

    fig.update_xaxes(title_text="Month/Year")

    fig.update_yaxes(title_text="sunspots")

    return fig


def plot_tsice_with_observed_features(explanation, feature_per_row=2):
    df = pd.DataFrame(explanation["data_x"])
    n_row = int(np.ceil(len(explanation["feature_names"]) / feature_per_row))
    feat_values = np.array(explanation["feature_values"])

    spec = [[{} for _ in range(feature_per_row)] for _ in range(n_row)]

    fig = make_subplots(n_row, feature_per_row, specs=spec)

    row_id = 1
    col_id = 1
    showlegend = True
    for i, feat in enumerate(explanation["feature_names"]):
        x_feat = feat_values[i, :, 0]
        trend_fit = LinearRegression()
        trend_line = trend_fit.fit(x_feat.reshape(-1, 1), explanation["signed_impact"])
        x_trend = np.linspace(min(x_feat), max(x_feat), 101)
        y_trend = trend_line.predict(x_trend[..., np.newaxis])

        fig.add_trace(
            go.Scatter(
                x=x_feat,
                y=explanation["signed_impact"],
                mode="markers",
                showlegend=False,
            ),
            row=row_id,
            col=col_id,
        )

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                line=dict(color="green"),
                mode="lines",
                name="correlation between forecast and observed feature",
                showlegend=showlegend,
            ),
            row=row_id,
            col=col_id,
        )
        current_value = explanation["current_feature_values"][i][0]
        reference_line = go.Scatter(
            x=[current_value, current_value],
            y=[
                np.min(explanation["signed_impact"]) - 1,
                np.max(explanation["signed_impact"]) + 1,
            ],
            mode="lines",
            line=go.scatter.Line(color="firebrick", dash="dash"),
            showlegend=showlegend,
            name="current value",
        )
        fig.add_trace(reference_line, row=row_id, col=col_id)

        fig.update_xaxes(title=f"<b>{feat}<b>", row=row_id, col=col_id)
        fig.update_yaxes(title=f"<b>&#916; forecast<b>", row=row_id, col=col_id)

        showlegend = False
        if col_id == feature_per_row:
            col_id = 1
            row_id += 1
        else:
            col_id += 1

    fig.update_layout(
        title="<b>Impact of Derived Variable On The Forecast</b>",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def plot_ts(df, df_timestamps, df_timestamp_name, df_targets, df_description):
    template = "plotly_white"
    traces = {}
    for target in df_targets:
        traces[target] = go.Scatter(
            name=target,
            x=df_timestamps,
            y=df[target],
            mode="lines",
            line=dict(color="rgb(0,0,0)"),
        )

    for target in df_targets:
        fig = go.Figure(data=[traces[target]])

        fig.update_layout(template=template)
        fig.update_layout(showlegend=False)
        fig.update_layout(hovermode="x")
        fig.update_layout(title_text="[target] %s" % (target))

        fig.update_xaxes(title_text=df_timestamp_name)
        fig.update_layout(xaxis_range=None)
        fig.update_yaxes(title_text=target)

        annotations = []
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=-0.02,
                y=-0.6,
                text=df_description,
                font=dict(family="Arial", size=10, color="rgb(150,150,150)"),
                showarrow=False,
            )
        )

        fig.update_layout(annotations=annotations)
    return fig
