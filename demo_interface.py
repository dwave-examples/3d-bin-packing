# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file stores the Dash HTML layout for the app."""
from __future__ import annotations

from dash import dcc, html

from demo_configs import (
    BIN_DIM,
    CASE_DIM,
    CHECKLIST,
    DESCRIPTION,
    DROPDOWN,
    MAIN_HEADER,
    NUM_BINS,
    NUM_CASES,
    RADIO,
    SOLVER_TIME,
    THEME_COLOR_SECONDARY,
    THUMBNAIL,
)
from src.demo_enums import SolverType


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configerations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label),
            dcc.Slider(
                id=id,
                className="slider",
                **config,
                marks={
                    config["min"]: str(config["min"]),
                    config["max"]: str(config["max"]),
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
        ],
    )


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=options[0]["value"],
                clearable=False,
                searchable=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list, values: list, inline: bool = True) -> html.Div:
    """Checklist element for option selection.

    Args:
        label: The title that goes above the checklist.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the checklist.
        inline: Whether the options of the checklist are displayed beside or below each other.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            html.Label(label),
            dcc.Checklist(
                id=id,
                className=f"checklist{' checklist--inline' if inline else ''}",
                inline=inline,
                options=options,
                value=values,
            ),
        ],
    )


def radio(label: str, id: str, options: list, value: int, inline: bool = True) -> html.Div:
    """Radio element for option selection.

    Args:
        label: The title that goes above the radio.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The value of the radio that should be preselected.
        inline: Whether the options are displayed beside or below each other.
    """
    return html.Div(
        className="radio-wrapper",
        children=[
            html.Label(label),
            dcc.RadioItems(
                id=id,
                className=f"radio{' radio--inline' if inline else ''}",
                inline=inline,
                options=options,
                value=value,
            ),
        ],
    )


def generate_options(options_list: list) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    return [{"label": label, "value": i} for i, label in enumerate(options_list)]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    dropdown_options = generate_options(DROPDOWN)
    checklist_options = generate_options(CHECKLIST)
    radio_options = generate_options(RADIO)

    solver_options = [
        {"label": solver_type.label, "value": solver_type.value} for solver_type in SolverType
    ]

    return html.Div(
        className="settings",
        children=[
            slider(
                "Number of Bins",
                "num-bins",
                NUM_BINS,
            ),
            slider(
                "Number of Cases",
                "num-cases",
                NUM_CASES,
            ),
            slider(
                "Case Dimension Range",
                "case-dim",
                CASE_DIM,
            ),
            html.Label("Bin Dimensions"),
            html.Div(
                [
                    html.Div([
                        html.Label("Length"),
                        dcc.Input(
                            id="bin-length",
                            type="number",
                            **BIN_DIM,
                        ),
                    ]),
                    html.Div([
                        html.Label("Width"),
                        dcc.Input(
                            id="bin-width",
                            type="number",
                            **BIN_DIM,
                        ),
                    ]),
                    html.Div([
                        html.Label("Height"),
                        dcc.Input(
                            id="bin-height",
                            type="number",
                            **BIN_DIM,
                        ),
                    ])
                ],
                className="display-flex-settings"
            ),
            
            # dropdown(
            #     "Example Dropdown",
            #     "dropdown",
            #     sorted(dropdown_options, key=lambda op: op["value"]),
            # ),
            # checklist(
            #     "Example Checklist",
            #     "checklist",
            #     sorted(checklist_options, key=lambda op: op["value"]),
            #     [0],
            # ),
            # radio(
            #     "Example Radio",
            #     "radio",
            #     sorted(radio_options, key=lambda op: op["value"]),
            #     0,
            # ),
            dropdown(
                "Solver",
                "solver-type-select",
                sorted(solver_options, key=lambda op: op["value"]),
            ),
            html.Label("Solver Time Limit (seconds)"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button(id="run-button", children="Run Optimization", n_clicks=0, disabled=False),
            html.Button(
                id="cancel-button",
                children="Cancel Optimization",
                n_clicks=0,
                className="display-none",
            ),
        ],
    )


def generate_problem_details_table_rows(solver: str, time_limit: int) -> list[html.Tr]:
    """Generates table rows for the problem details table.

    Args:
        solver: The solver used for optimization.
        time_limit: The solver time limit.

    Returns:
        list[html.Tr]: List of rows for the problem details table.
    """

    table_rows = (
        ("Solver:", solver, "Time Limit:", f"{time_limit}s"),
        ### Add more table rows here. Each tuple is a row in the table.
    )

    return [html.Tr([html.Td(cell) for cell in row]) for row in table_rows]


def problem_details(index: int) -> html.Div:
    """Generate the problem details section.

    Args:
        index: Unique element id to differentiate matching elements.
            Must be different from left column collapse button.

    Returns:
        html.Div: Div containing a collapsable table.
    """
    return html.Div(
        id={"type": "to-collapse-class", "index": index},
        className="details-collapse-wrapper collapsed",
        children=[
            # Problem details collapsible button and header
            html.Button(
                id={"type": "collapse-trigger", "index": index},
                className="details-collapse",
                children=[
                    html.H5("Problem Details"),
                    html.Div(className="collapse-arrow"),
                ],
            ),
            html.Div(
                className="details-to-collapse",
                children=[
                    html.Table(
                        className="solution-stats-table",
                        children=[
                            # Problem details table header (optional)
                            html.Thead(
                                [
                                    html.Tr(
                                        [
                                            html.Th(
                                                colSpan=2,
                                                children=["Problem Specifics"],
                                            ),
                                            html.Th(
                                                colSpan=2,
                                                children=["Run Time"],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            # A Dash callback function will generate content in Tbody
                            html.Tbody(id="problem-details"),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_interface():
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            # Header brand banner
            html.Div(className="banner", children=[html.Img(src=THUMBNAIL)]),
            # Settings and results columns
            html.Div(
                className="columns-main",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.H1(MAIN_HEADER),
                                            html.P(DESCRIPTION),
                                            generate_settings_form(),
                                            generate_run_buttons(),
                                        ],
                                    )
                                ],
                            ),
                            # Left column collapse button
                            html.Div(
                                html.Button(
                                    id={"type": "collapse-trigger", "index": 0},
                                    className="left-column-collapse",
                                    children=[html.Div(className="collapse-arrow")],
                                ),
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="input-tab",
                                mobile_breakpoint=0,
                                children=[
                                    dcc.Tab(
                                        label="Input",
                                        id="input-tab",
                                        value="input-tab",  # used for switching tabs programatically
                                        className="tab",
                                        children=[
                                            dcc.Loading(
                                                parent_className="input",
                                                type="circle",
                                                color=THEME_COLOR_SECONDARY,
                                                # A Dash callback (in app.py) will generate content in the Div below
                                                children=html.Div(id="input"),
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Results",
                                        id="results-tab",
                                        className="tab",
                                        disabled=True,
                                        children=[
                                            html.Div(
                                                className="tab-content-results",
                                                children=[
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR_SECONDARY,
                                                        # A Dash callback (in app.py) will generate content in the Div below
                                                        children=html.Div(id="results"),
                                                    ),
                                                    # Problem details dropdown
                                                    html.Div([html.Hr(), problem_details(1)]),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
