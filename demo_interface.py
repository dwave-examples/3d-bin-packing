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
from enum import EnumMeta

from dash import dcc, html
import dash_mantine_components as dmc

from demo_configs import (
    COLOR_BY_CASE,
    BIN_HEIGHT,
    BIN_LENGTH,
    BIN_WIDTH,
    CASE_DIM,
    DESCRIPTION,
    MAIN_HEADER,
    NUM_BINS,
    NUM_CASES,
    SOLVER_TIME,
    THUMBNAIL,
)
from src.demo_enums import ProblemType, ScenarioType, SolverType
from utils import TABLE_HEADERS

THEME_COLOR = "#2d4376"


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
            ),
        ],
    )


def range_slider(label: str, id: str, config: dict) -> html.Div:
    """Range slider element for value selection.

    Args:
        label: The title that goes above the range slider.
        id: A unique selector for this element.
        config: A dictionary of range slider configurations, see dmc.RangeSlider Dash Mantine docs.
    """
    return html.Div(
        className="rangeslider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.RangeSlider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbFromLabel=f"{label} slider start",
                thumbToLabel=f"{label} slider end",
                color=THEME_COLOR,
            )
        ]
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
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def checkbox(label: str, id: str, checked: bool) -> html.Div:
    """Checkbox element.

    Args:
        label: The title that goes above the checkbox.
        id: A unique selector for this element.
        checked: Whether the checkbox is checked or not.
    """
    return html.Div(
        className="checkbox-wrapper",
        children=[
            dmc.Checkbox(
                id=id,
                label=label,
                checked=checked,
                color=THEME_COLOR,
            )
        ],
    )


def generate_options(options: list | EnumMeta) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    if isinstance(options, EnumMeta):
        return [
            {"label": option.label, "value": f"{option.value}"} for option in options
        ]

    return [{"label": option, "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    problem_type_options = generate_options(ProblemType)
    solver_options = generate_options(SolverType)
    scenario_options = generate_options(ScenarioType)

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Problem Type",
                "problem-type",
                sorted(problem_type_options, key=lambda op: op["value"]),
            ),
            html.Div(
                [
                    html.Label("Problem File", htmlFor="input-file"),
                    dcc.Upload(
                        id="input-file",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select a File"), html.Div(id="filename")]
                        ),
                    ),
                ],
                id="uploaded-settings",
                className="display-none",
            ),
            html.Div(
                [
                    slider(
                        "Number of Bins",
                        "num-bins",
                        NUM_BINS,
                    ),
                    html.Label("Bin Dimensions", htmlFor="display-flex-settings"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Length", htmlFor="bin-length"),
                                    dmc.NumberInput(
                                        id="bin-length",
                                        **BIN_LENGTH,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Width", htmlFor="bin-width"),
                                    dmc.NumberInput(
                                        id="bin-width",
                                        **BIN_WIDTH,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Height", htmlFor="bin-height"),
                                    dmc.NumberInput(
                                        id="bin-height",
                                        **BIN_HEIGHT,
                                    ),
                                ]
                            ),
                        ],
                        className="display-flex-settings",
                        id="display-flex-settings",
                    ),
                    slider(
                        "Number of Cases",
                        "num-cases",
                        NUM_CASES,
                    ),
                    range_slider(
                        "Case Dimension Bounds",
                        "case-dim",
                        CASE_DIM,
                    ),
                ],
                id={"type": "generated-settings", "index": 0},
            ),
            html.Div(
                [
                    dropdown(
                        "Scenario",
                        "scenario-select",
                        sorted(scenario_options, key=lambda op: op["value"]),
                    ),
                ],
                className="display-none",
                id="scenario-settings",
            ),
            dropdown(
                "Solver",
                "solver-type-select",
                sorted(solver_options, key=lambda op: op["value"]),
            ),
            html.Label("Solver Time Limit (seconds)", htmlFor="solver-time-limit"),
            dmc.NumberInput(
                id="solver-time-limit",
                **SOLVER_TIME,
            ),
            html.Label("Write Solution to File", htmlFor="save-solution"),
            dmc.TextInput(id="save-solution", placeholder="File Name"),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Optimization", id="run-button", className="button"),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def generate_table(table_data: dict) -> list[html.Thead, html.Tbody]:
    """Generates the input table.

    Args:
        table_data: Dictionary of lists of input data.

    Returns:
        list: The table head and table body of the results table.
    """
    body = [
        [table_data[table_header][i] for table_header in TABLE_HEADERS]
        for i in range(len(table_data[TABLE_HEADERS[0]]))
    ]
    return [
        html.Thead([html.Tr([html.Th(header) for header in TABLE_HEADERS])]),
        html.Tbody(generate_table_rows(body)),
    ]


def generate_table_rows(table_data: list[list]) -> list[html.Tr]:
    """Generates table rows.

    Args:
        table_data: A list of lists to display in table rows.

    Returns:
        list[html.Tr]: List of rows.
    """

    return [html.Tr([html.Td(cell) for cell in row]) for row in table_data]


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
                    html.H5("Model Details"),
                    html.Div(className="collapse-arrow"),
                ],
                **{"aria-expanded": "true"},
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
                                                colSpan=3,
                                                children=["Variables"],
                                            ),
                                            html.Th(
                                                colSpan=3,
                                                children=["Constraints"],
                                            ),
                                            html.Th(
                                                colSpan=3,
                                                children=["Sensitivity"],
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
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="problem-data-store"),
            dcc.Store(id="max-bins-store"),
            dcc.Store(id="bin-dimensions-store"),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
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
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="input-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab("Input", value="input-tab"),
                                                            dmc.TabsTab(
                                                                "Results",
                                                                value="results-tab",
                                                                id="results-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="input-tab",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div(
                                                        className="input",
                                                        children=[
                                                            html.Div(
                                                                html.Table(
                                                                    id="input",
                                                                    # add children dynamically using 'generate_table'
                                                                ),
                                                                tabIndex="14",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.H5(
                                                                        [
                                                                            "Maximum bins: ",
                                                                            html.Span(id="max-bins"),
                                                                        ]
                                                                    ),
                                                                    html.H5(
                                                                        [
                                                                            "Bin dimensions: ",
                                                                            html.Span(id="bin-dims"),
                                                                            " (L*W*H)",
                                                                        ]
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            html.Label(
                                                                                "Save Input Data to File",
                                                                                htmlFor="save-input-filename",
                                                                            ),
                                                                            html.Div(
                                                                                [
                                                                                    html.Div([
                                                                                        dmc.TextInput(
                                                                                            id="save-input-filename",
                                                                                            placeholder="File Name",
                                                                                        ),
                                                                                        html.Button(
                                                                                            id="save-input-button",
                                                                                            children="Save",
                                                                                            n_clicks=0,
                                                                                            className="button button-small",
                                                                                        ),
                                                                                    ], className="save-input-wrapper"),
                                                                                    html.P(
                                                                                        "Saved to input folder",
                                                                                        className="display-none",
                                                                                        id="saved",
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                        ],
                                                                        id={
                                                                            "type": "generated-settings",
                                                                            "index": 1,
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        tabIndex="13",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    checkbox(
                                                        COLOR_BY_CASE,
                                                        "color-by-case",
                                                        False,
                                                    ),
                                                    dcc.Loading(
                                                        parent_className="results",
                                                        type="circle",
                                                        color=THEME_COLOR,
                                                        children=html.Div(
                                                            dcc.Graph(
                                                                id="results",
                                                                responsive=True,
                                                                config={"displayModeBar": False},
                                                            ),
                                                            className="graph",
                                                        ),
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
