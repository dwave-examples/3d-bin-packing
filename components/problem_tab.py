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
     Output('info-card', 'color'),
     Output('models', 'data')],
    Input('solve-button', 'n_clicks'),
    [State(f"{setting['id']}", 'value') for setting in settings] +
    [State(f"{setting['id']}", 'value') for setting in settings_solve] +
    [State('models', 'data')],
    prevent_initial_callback=True
)
def solve(nc, *args):
    models = args[-1]
    args = args[:-1]
    if nc > 0:
        figure, message, feasible, model = _solve(*args)
        models.append(model)
        return figure, message, feasible, models
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
        data = generate_data(**kwargs)
    else:
        data = read_instance(kwargs['data_filepath'])

    path = f'input/{file_name(kwargs)}.json'
    result = main(data=data, solution_file=path, **kwargs)
    if result['feasible']:
        result['figure'].update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        kwargs['solution'] = result['solution']
        return dcc.Graph(figure=result['figure'], style={'height': '700px'}), \
               "Feasible Solution Found", 'success', kwargs
    elif result['suitable']:
        return '', \
               "No Feasible Solution Found", 'danger', kwargs
    else:
        return '', "MIP cannot support quadratic interactions", 'danger', kwargs


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


@app.callback(
    Output("time_limit", "value"),
    [Input("time_limit", "value"), Input("solver", "value")],

)
def adjust_time_limit(tl, solver):
    if tl < 5 and solver == 'Constrained Quadratic Model':
        return 5
    return dash.no_update


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


def file_name(p):
    from itertools import chain
    p = dict(p)
    if p['input_type'] == 'Random':
        p.pop('data_filepath')
    else:
        for key in ['num_bins', 'num_cases', 'case_size_range_min',
                    'case_size_range_max', 'bin_dimensions', 'bin_length',
                    'bin_width', 'bin_height']:
            p.pop(key)
    for key in p:
        if isinstance(p[key], str):
            for c in list('/.- '):
                p[key] = p[key].replace(c, '_')
            for c in list('()'):
                p[key] = p[key].replace(c, '')
    name = '_'.join(map(str, chain(*p.items())))
    return name
