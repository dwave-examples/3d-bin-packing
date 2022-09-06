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

from dash import dcc, html, Input, Output
from .overview_tab import build_overview_tab
from .problem_tab import build_problem_tab
from .benchmark_tab import build_benchmark_tab
from .app import app
from .content import content

_tabs = {
    'name': content['tabs']
}


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id=f"tab-{i}",
                        label=f"{name}",
                        value=f"tab{i}",
                        className=f"custom-tab",
                        selected_className=f"custom-tab--selected",
                        disabled_className="custom-tab--disabled"
                    )
                    for i, name in enumerate(_tabs['name'])
                ],
            )
        ],
    )


@app.callback(
    Output("app-content", "children"),
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    if tab_switch == "tab0":
        return build_overview_tab()
    if tab_switch == "tab1":
        return build_problem_tab()
    if tab_switch == "tab2":
        return build_benchmark_tab()
