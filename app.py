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

from components import build_banner
from components import build_tabs
from components import app


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Store(id='models', data=[]),
        dcc.Interval(
            id="interval-component",
            interval=1000,
            n_intervals=50,
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                html.Div(id="app-content"),
            ],
        ),
    ],
)


if __name__ == "__main__":
    app.run_server(debug=True, port=8501)