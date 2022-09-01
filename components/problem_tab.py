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

import dash
import dash_bootstrap_components as dbc
import numpy as np
from functools import lru_cache
from dash import html, dcc, Input, Output, State
from .utils import selector
from .settings import settings, settings_solve
from .app import app
from .packing3d import main
from .bin_packing_utils import read_instance

button_style = {'background-color': 'white',
                'color': 'black'}

div_style = {'margin-top': '3%', 'margin-left': '15%'}

displays = [{'display': s.get('display', 'block')} for s in settings[1:]]
always_blocked = [s['id'] in ('case_size_range_min', 'case_size_range_max',
                              'bin_length', 'bin_width', 'bin_height')
                  for s in settings[1:]]


def build_problem_tab():
    return html.Div(children=[
        dbc.Card(children=[
            dbc.Row([
                dbc.Col(
                    children=[
                        html.Div([selector(**setting)
                                  for setting in settings],
                                 style={'margin': '10px 0px 0px 10px'}),
                        html.Div([selector(**setting)
                                  for setting in settings_solve],
                                 style={'margin': '20px 0px 0px 10px'}),
                        html.Div(html.Button('solve', id='solve-button',
                                             style=button_style,
                                             n_clicks=0),
                                 style={'margin': '20px 0px 0px 10px'}),
                        dbc.Card(children='Press Solve...', id='info-card',
                                 style={'color': 'black',
                                        'margin': '20px 0px 0px 10px'})
                    ]
                    , width=2),
                dbc.Col(html.Div(id='output', style={'height': '1000px'}),
                        width=10),
            ]),
        ], color='#1E2130', style={'margin': '10px 0px 0px 0px'}),
    ], style={'width': '98%', 'margin-left': '1%', 'height': '2000px'})


@app.callback(
    [Output('output', 'children'),
     Output('info-card', 'children'),
     Output('info-card', 'color')],
    Input('solve-button', 'n_clicks'),
    [State(f"{setting['id']}", 'value') for setting in settings] +
    [State(f"{setting['id']}", 'value') for setting in settings_solve],
    prevent_initial_callback=True
)
def solve(nc, *args):
    if nc > 0:
        return _solve(*args)
    return dash.no_update


@lru_cache()
def _solve(*args):
    ids = [setting['id'] for setting in settings] + [setting['id'] for setting
                                                     in settings_solve]
    kwargs = {_id: value for _id, value in zip(ids, args)}

    kwargs['bin_length'], kwargs['bin_width'], kwargs['bin_height'] = list(
        map(int, kwargs['bin_dimensions'].split('x'))
    )
    kwargs['use_cqm_solver'] = 'Constrained Quadratic Model' == kwargs['solver']
    if kwargs['input_type'] == 'Random':
        print("Using random data")
        data = generate_data(**kwargs)
    else:
        print("Using a file")
        data = read_instance(kwargs['data_filepath'])
    print(data)
    fig, found_feasible, suitable = main(data=data, **kwargs)
    if found_feasible:
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return dcc.Graph(figure=fig, style={'height': '700px'}), \
               "Feasible Solution Found", 'success'
    elif suitable:
        return '', \
               "No Feasible Solution Found", 'danger'
    else:
        return '', "MIP cannot support quadratic interactions", 'danger'


@app.callback(
    [Output(f"div_{setting['id']}", 'style') for setting in settings[1:]],
    Input('input_type', 'value'),
    prevent_initial_callback=True
)
def update_inputs(value):
    if value == 'Random':
        return displays
    return [{'display': 'none' if d['display'] == 'block' or ab else 'block'}
            for ab, d in zip(always_blocked, displays)]


def generate_data(num_bins, num_cases,
                  bin_length, bin_width, bin_height,
                  case_size_range_min,
                  case_size_range_max, **kwargs):
    seed = 111
    rng = np.random.default_rng(seed)
    data = {
        "num_bins": num_bins,
        "bin_dimensions": [bin_length, bin_width, bin_height],
        "case_length": rng.integers(
            case_size_range_min, case_size_range_max,
            num_cases, endpoint=True
        ),
        "case_width": rng.integers(
            case_size_range_min, case_size_range_max,
            num_cases, endpoint=True
        ),
        "case_height": rng.integers(
            case_size_range_min, case_size_range_max,
            num_cases, endpoint=True
        ),
    }
    case_dimensions = np.vstack(
        [data["case_length"], data["case_width"], data["case_height"]]
    )
    unique_dimensions, data["quantity"] = np.unique(case_dimensions,
                                                    axis=1,
                                                    return_counts=True)

    data["case_length"] = unique_dimensions[0, :]
    data["case_width"] = unique_dimensions[1, :]
    data["case_height"] = unique_dimensions[2, :]

    data["case_ids"] = np.array(range(len(data["quantity"])))
    return data
