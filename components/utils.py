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

from dash import dcc, html
import dash_bootstrap_components as dbc


def named_template(name, element, **kwargs):
    style = {'margin': '1px 1px 1px 1px',
             'display': kwargs.get('display', 'block')}
    if 'display' in kwargs:
        kwargs.pop('display')
    return html.Div([
        dbc.Row(html.Div(f"{name}", style={'text-align': 'left',
                                           'font-size': 12,
                                           'margin': '1px 1px 1px 1px',
                                           })),
        dbc.Row(html.Div(element)),
    ], style=style, id=f"div_{kwargs['id']}",
    )


def named_input(name='Dummy', id='', value='', type='text', **kwargs):
    min_value = kwargs.get('min', None)
    return named_template(
        name,
        dcc.Input(id=id, value=value, type=type, style={'width': '100%'},
                  min=min_value,
                  debounce=True), id=id,
        **kwargs
    )


def named_dropdown(name='Dummy', id='dummy',
                   value='', options=[], **kwargs):
    return named_template(
        name,
        dcc.Dropdown(
            id=id,
            clearable=False,
            value=value,
            options=options,
            style={'width': '100%', 'color': 'black'},
        ), id=id, **kwargs
    )


def selector(**kwargs):
    if 'value' not in kwargs:
        raise ValueError

    input_type = kwargs.pop('type')
    if input_type == 'menu':
        return named_dropdown(**kwargs)
    kwargs.pop('options')
    if input_type == 'text':
        return named_input(**kwargs, type=input_type)
    if input_type == 'number':
        return named_input(**kwargs, type=input_type)
