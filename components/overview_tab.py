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

from dash import html
import dash_bootstrap_components as dbc
from .content import content

content_slides = content['overview_slides']


def build_overview_tab():
    return [html.Div(children=[
        html.Div(switch_pages()),
    ], style={'textAlign': 'center'})]


def switch_pages():
    el = [html.Img(
        src=content_slides[0],
        className='overview-image'),
        html.H6(content['content'],
                style={'color': 'black', 'margin': '5px 5px 5px 5px'})
    ]
    el = dbc.Card(el, style={'width': '70%', 'margin-left': '15%',
                             'margin-top': '2.5%'})
    return el
