from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np


def plot_ice_explanation(explanation, title):
    feature_names = list(explanation.keys())
    
    fig = plt.figure(layout='constrained', figsize=(15,10))
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios= [1, int(len(feature_names)/3)], top=0.9)
    gs2 = gridspec.GridSpecFromSubplotSpec(int(len(feature_names)/3), 3, subplot_spec=gs0[1])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[4.5, 10, 4.5])

    for i, feature_col in enumerate(feature_names):
        if i > 0:
            gs = gs2[i-1]
        else:
            gs = gs1[1]


        ax = fig.add_subplot(gs)
        delta_avg = 0
        delta_ice_values = []
        for i in range(len(explanation[feature_col])):
            delta_ice = explanation[feature_col][i]["ice_value"] - explanation[feature_col][i]["ice_value"][0]
            delta_ice_values.append(delta_ice)
            ax.plot(explanation[feature_col][i]["feature_value"], delta_ice, color='b', alpha=0.2)


        delta_ice_values = np.asarray(delta_ice_values)
        # average ice value across test instances
        delta_avg = np.mean(delta_ice_values, axis=0)
        ax.plot(explanation[feature_col][0]["feature_value"], delta_avg, color='r')  

        ax.set_title("{}".format(feature_col))

        if i==0:
            ax.set_ylabel('Model Prediction')
            ax.set_xlabel('Feature Value')

    fig.suptitle(title, fontsize='x-large')
    plt.show()

def plot_gce_explanation(
    explanation,
    plot_width: int = 250,
    plot_height: int = 250,
    plot_bgcolor: str = "white",
    plot_line_width: int = 2,
    plot_instance_size: int = 15,
    plot_instance_color: str = "firebrick",
    plot_instance_width: int = 4,
    plot_contour_coloring: str = "heatmap",
    plot_contour_color: Union[str, List[Tuple[float, str]]] = "Portland",
    renderer="notebook",
    title=None,
    **kwargs,
):

    exp_data = explanation
    feat_dict = {k: len(exp_data[k].keys()) for k in exp_data['selected_features']}
    features = sorted(feat_dict, key=lambda l: feat_dict[l])
    n_feat = len(features)

    specs = [
        [{} if i <= j else None for j in range(n_feat - 1)] for i in range(n_feat - 1)
    ]

    fig = make_subplots(
        rows=n_feat - 1,
        cols=n_feat - 1,
        specs=specs,
        shared_xaxes="columns",
        shared_yaxes="rows",
        column_titles=features[1:],
        row_titles=features[:-1],
    )

    for x_i in range(n_feat):
        for y_i in range(n_feat):
            if y_i < x_i:
                x_feat = features[x_i]
                y_feat = features[y_i]
                z = exp_data[x_feat][y_feat]["gce_values"]
                x_g = exp_data[x_feat][y_feat]["x_grid"]
                y_g = exp_data[x_feat][y_feat]["y_grid"]
                fig.add_trace(
                    go.Contour(
                        z=z,
                        x=x_g,
                        y=y_g,
                        connectgaps=True,
                        line_smoothing=0.5,
                        contours_coloring=plot_contour_coloring,
                        contours_showlabels=True,
                        line_width=plot_line_width,
                        coloraxis="coloraxis1",
                        hovertemplate="<b>"
                        + str(x_feat)
                        + "</b>: %{x:.2f}<br>"
                        + "<b>"
                        + str(y_feat)
                        + "</b>: %{y:.2f}<br>"
                        + "<b>prediction</b>: %{z:.2f}<br><extra></extra>",
                    ),
                    row=y_i + 1,
                    col=x_i,
                )
                if "current_values" in exp_data[x_feat][y_feat]:
                    x = exp_data[x_feat][y_feat]["current_values"][x_feat]
                    y = exp_data[x_feat][y_feat]["current_values"][y_feat]
                    fig.add_trace(
                        go.Scatter(
                            mode="markers",
                            marker_symbol="x",
                            x=[x],
                            y=[y],
                            marker_color=plot_instance_color,
                            marker_line_color=plot_instance_color,
                            marker_size=plot_instance_size,
                            marker_line_width=plot_instance_width,
                            showlegend=False,
                            hovertemplate="{}: {:.2f}<br> {}: {:.2f}<extra></extra>".format(
                                x_feat, x, y_feat, y
                            ),
                        ),
                        row=y_i + 1,
                        col=x_i,
                    )

    fig.update_layout(
        height=(n_feat - 1) * plot_height,
        width=(n_feat - 1) * plot_width,
        plot_bgcolor=plot_bgcolor,
        coloraxis_autocolorscale=False,
        coloraxis_colorscale=plot_contour_color,
        title_text=title
        
    )
    return fig