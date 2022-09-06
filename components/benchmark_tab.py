# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import html, dcc, Input, Output
from .utils import selector
from .app import app
from .content import content
import plotly.graph_objs as go


def build_benchmark_tab():
    return html.Div(children=[
        dbc.Card(children=[
            dbc.Row([
                dbc.Col(
                    children=selector(
                        type='menu',
                        name='number of cases',
                        id='cases-menu',
                        value=10,
                        options=[10, 20, 30, 40]
                    )
                    , width=2),
                dbc.Col(html.Div(id='output-benchmark', style={'height': '500px'}),
                        width=10),
            ]),
        ], color='#1E2130', style={'margin': '10px 0px 0px 0px'}),
    ], style={'width': '98%', 'margin-left': '1%', 'height': '2000px'})


@app.callback(
    Output('output-benchmark', 'children'),
    [Input('cases-menu', 'value'),
     Input("app-tabs", "value")]
)
def benchmark_results(num_cases, tab):
    if tab == 'tab2':
        data = json.load(open(f'components/assets/benchmark_{num_cases}.json'))
        import pandas as pd
        data = pd.DataFrame(data)
        return plot_data(data)
    return dash.no_update


def plot_data(data):
    fig = go.Figure()
    color = content['benchmark_tab_trace_colors']
    for i, k in enumerate(data.keys()):
        x = np.array(data[k]['x'])
        ind = np.argsort(x)
        x = x[ind]
        y = np.array(data[k]['y'])[ind]
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                name=f'{k}',
                mode='markers+lines',
                marker=dict(
                    color=color[i],
                    size=10,
                )))

    fig.update_xaxes(type="log")
    fig.update_layout(xaxis=dict(title="Time (s)"),
                      yaxis=dict(title="Maximum Height (cm)"),
                      )

    return html.Div(children=[
        dbc.Card(
            dcc.Graph(figure=fig),
            className='card-img-top',
            color='#DBE6F3',
            style={'margin-left': '12.5%', 'width': '75%', 'margin-top': '2%'}
        )],
        style={'align': 'center', 'width': '75%', 'margin-left': '12.5%',
               'margin-top': '4%'})
