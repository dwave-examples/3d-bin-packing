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

from __future__ import annotations

import base64
import json
from typing import NamedTuple, Union

import dash
import numpy as np
import plotly.graph_objs as go
from dash import ALL, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from demo_configs import RANDOM_SEED
from demo_interface import generate_table, generate_table_rows
from packing3d import Bins, Cases, Variables, build_cqm, call_solver
from src.demo_enums import ProblemType, SolverType
from utils import (
    case_list_to_dict,
    get_cqm_stats,
    plot_cuboids,
    update_colors,
    write_input_data,
    write_solution_to_file,
)


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> str:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        str: The new class name of the thing to collapse.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes)
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed"


@dash.callback(
    Output({"type": "generated-settings", "index": ALL}, "className"),
    Output("uploaded-settings", "className"),
    Output("scenario-settings", "className"),
    inputs=[
        Input("problem-type", "value"),
        State({"type": "generated-settings", "index": ALL}, "children"),
    ],
)
def update_problem_type(
    problem_type: Union[ProblemType, int],
    gen_settings: list,
) -> tuple[list[str], str]:
    """Updates the visible settings when the Problem Type is changed.

    Args:
        problem_type: The ProblemType that was just selected from the dropdown.
        gen_settings: The settings for the `Generated` ProblemType.

    Returns:
        list[str]: The classe names for the settings for the `Generated` ProblemType.
        str: The class name for the `Uploaded` ProblemType.
    """
    if problem_type is ProblemType.FILE.value:
        return ["display-none"] * len(gen_settings), "", "display-none"

    if problem_type is ProblemType.SCENARIO.value:
        return ["display-none"] * len(gen_settings), "display-none", ""

    return [""] * len(gen_settings), "display-none", "display-none"


@dash.callback(
    Output("input", "children"),
    Output("max-bins", "children"),
    Output("bin-dims", "children"),
    Output("problem-data-store", "data"),
    Output("saved", "className"),
    inputs=[
        Input("problem-type", "value"),
        Input("num-cases", "value"),
        Input("case-dim", "value"),
        Input("num-bins", "value"),
        Input("bin-length", "value"),
        Input("bin-width", "value"),
        Input("bin-height", "value"),
    ],
)
def generate_data(
    problem_type: Union[ProblemType, int],
    num_cases: int,
    case_size_range: list[int],
    num_bins: int,
    bin_length: int,
    bin_width: int,
    bin_height: int,
) -> tuple[list, int, str, dict, str]:
    """Updates the input table when ProblemType is `Generated` and any relevant settings have been
    changed.

    Args:
        problem_type: The input problem type. Either Generated or Uploaded.
        num_cases: The value of the number of cases setting.
        case_size_range: The values of the case size range setting.
        num_bins: The current value of the number of bins setting.
        bin_length: The current value of the bin length setting.
        bin_width: The current value of the bin width setting.
        bin_height: The current value of the bin height setting.

    Returns:
        input: The input table.
        max-bins: The maximum bins to display in the input UI.
        bin-dimensions: The bin dimension string to display in the UI.
        problem-data-store: The data that was generated for the table.
        saved: The class name for the `Saved!` feedback.
    """
    if ProblemType(problem_type) is not ProblemType.GENERATED:
        raise PreventUpdate

    rng = np.random.default_rng(RANDOM_SEED)

    case_dimensions = np.array(
        [
            rng.integers(case_size_range[0], case_size_range[1], num_cases, endpoint=True)
            for i in range(3)
        ]
    )

    # Determine quantities and case_ids
    unique_dimensions, quantity = np.unique(
        case_dimensions, axis=1, return_counts=True
    )

    problem_data = {
        "Case ID": np.arange(len(quantity)),
        "Quantity": quantity,
        "Length": unique_dimensions[0],
        "Width": unique_dimensions[1],
        "Height": unique_dimensions[2],
        "num_bins": num_bins,
        "bin_dimensions": [bin_length, bin_width, bin_height],
    }

    return (
        generate_table(problem_data),
        num_bins,
        f"{bin_length} * {bin_width} * {bin_height}",
        problem_data,
        "display-none"
    )


@dash.callback(
    Output("input", "children", allow_duplicate=True),
    Output("max-bins", "children", allow_duplicate=True),
    Output("bin-dims", "children", allow_duplicate=True),
    Output("problem-data-store", "data", allow_duplicate=True),
    Output("saved", "className", allow_duplicate=True),
    inputs=[
        Input("problem-type", "value"),
        Input("scenario-select", "value"),
    ],
    prevent_initial_call=True,
)
def load_scenario(
    problem_type: Union[ProblemType, int],
    scenario: int,
) -> tuple[list, int, str, dict, str]:
    """Updates the input table when ProblemType is `Scenario` has changed.

    Args:
        problem_type: The input problem type. Either Generated or Uploaded.
        scenario_select: The current value of the scenario dropdown.

    Returns:
        input: The input table.
        max-bins: The maximum bins to display in the input UI.
        bin-dimensions: The bin dimension string to display in the UI.
        problem-data-store: The data that was generated for the table.
        saved: The class name for the `Saved!` feedback.
    """
    if ProblemType(problem_type) is not ProblemType.SCENARIO:
        raise PreventUpdate

    scenarios = json.load(open("./src/data/scenarios.json", "r"))

    scenario_data = scenarios[str(scenario)]

    bin_length, bin_width, bin_height = scenario_data["bin_dimensions"]

    return (
        generate_table(scenario_data),
        scenario_data["num_bins"],
        f"{bin_length} * {bin_width} * {bin_height}",
        scenario_data,
        "display-none"
    )


class ReadInputFileReturn(NamedTuple):
    """Return type for the ``read_input_file`` callback function."""

    table_input: list = dash.no_update
    max_bins: int = dash.no_update
    bin_dimensions: str = dash.no_update
    filename: str = dash.no_update
    problem_data_store: dict = dash.no_update


@dash.callback(
    Output("input", "children", allow_duplicate=True),
    Output("max-bins", "children", allow_duplicate=True),
    Output("bin-dims", "children", allow_duplicate=True),
    Output("filename", "children"),
    Output("problem-data-store", "data", allow_duplicate=True),
    inputs=[
        Input("input-file", "contents"),
        Input("problem-type", "value"),
        State("input-file", "filename"),
    ],
    prevent_initial_call=True,
)
def read_input_file(
    file_contents: str,
    problem_type: Union[ProblemType, int],
    filename: str,
) -> ReadInputFileReturn:
    """Reads input file and displays data in a table.

    Args:
        file_contents: The encoded contents of the uploaded input file.
        problem_type: The input problem type. Either Generated or Uploaded.
        filename: The name of the uploaded file.

    Returns:
        A NamedTuple (ReadInputFileReturn) with the following parameters:
            table_input: The input table containing problem data from the file.
            max_bins: The maximum bins to display in the input UI.
            bin_dimensions: The bin dimension string to display in the UI.
            filename: The name of the file that was uploaded to display in the UI.
            problem_data_store: The value to update the table data store.
    """
    if ProblemType(problem_type) is not ProblemType.FILE:
        raise PreventUpdate

    if file_contents is not None:
        decoded = base64.b64decode(file_contents)

        try:
            lines = decoded.decode("ISO-8859-1").splitlines()

            num_bins = int(lines[0].split(":")[1].strip())
            bin_length, bin_width, bin_height = map(int, lines[1].split(":")[1].split())

            case_data = []
            for line in lines[5:]:
                if line.strip():
                    case_data.append(list(map(int, line.split())))

            problem_data = case_list_to_dict(
                case_data, num_bins, [bin_length, bin_width, bin_height]
            )

            return ReadInputFileReturn(
                table_input=generate_table(problem_data),
                max_bins=num_bins,
                bin_dimensions=f"{bin_length} * {bin_width} * {bin_height}",
                filename=filename,
                problem_data_store=problem_data,
            )

        except Exception as e:
            print(e)
            return ReadInputFileReturn(filename="There was an error processing this file.")

    raise PreventUpdate


@dash.callback(
    Output("saved", "className", allow_duplicate=True),
    inputs=[
        Input("save-input-button", "n_clicks"),
        State("save-input-filename", "value"),
        State("problem-data-store", "data"),
    ],
    prevent_initial_call=True,
)
def save_input_to_file(
    save_button: int,
    filename: str,
    problem_data: dict,
) -> str:
    """Saves input data to a text file when the `save-input-button` is clicked.

    Args:
        save_button: How many times the save to file button has been clicked.
        filename: The file name to save the input data to.
        problem_data: The data from the table of input values.

    Returns:
        str: The `Saved!` text class name.
    """
    write_input_data(problem_data, filename)

    return ""


@dash.callback(
    Output("results", "figure", allow_duplicate=True),
    inputs=[
        Input("checklist", "value"),
        State("results", "figure"),
    ],
    prevent_initial_call=True,
)
def update_graph_colors(
    checklist: list,
    fig: go.Figure,
) -> go.Figure:
    """Updates the colors of the figure when the value of the checklist changes.

    Args:
        checklist: A list of the current values of the checklist.
        fig: The current figure that is displayed.

    Returns:
        go.Figure: The updated figure.
    """
    return update_colors(fig, bool(checklist))


@dash.callback(
    Output("results", "figure"),
    Output("problem-details", "children"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-type-select", "value"),
        State("solver-time-limit", "value"),
        State("problem-data-store", "data"),
        State("checklist", "value"),
        State("save-solution", "value"),
    ],
    running=[
        (Output("cancel-button", "className"), "", "display-none"),  # Show/hide cancel button.
        (Output("run-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("problem-type", "disabled"), True, False),
        (Output("num-cases", "disabled"), True, False),
        (Output("case-dim", "disabled"), True, False),
        (Output("num-bins", "disabled"), True, False),
        (Output("bin-length", "disabled"), True, False),
        (Output("bin-width", "disabled"), True, False),
        (Output("bin-height", "disabled"), True, False),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    problem_data: dict,
    checklist: list,
    save_solution_filepath: str,
) -> tuple[go.Figure, list]:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the ``Run Optimization`` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and updates all relevant HTML
    components.

    Args:
        run_click: The number of times the run button has been clicked.
        solver_type: The value of the Solver form field.
        time_limit: The value of the Solver Time Limit form field.
        problem_data: The stored generated data.
        checklist: The current value of the checklist.
        save_solution_filepath: The filepath to save the solution to.

    Returns:
        fig: The results figure.
        problem_details_table: The table and information to display in the problem details table.
    """
    cases = Cases(problem_data)
    bins = Bins(problem_data, cases)
    vars = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(vars, bins, cases)

    best_feasible = call_solver(cqm, time_limit, solver_type is SolverType.CQM.value)

    if save_solution_filepath is not None:
        write_solution_to_file(
            save_solution_filepath, cqm, vars, best_feasible, cases, bins, effective_dimensions
        )

    fig = plot_cuboids(best_feasible, vars, cases, bins, effective_dimensions, bool(checklist))

    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_table_rows(get_cqm_stats(cqm))

    return fig, problem_details_table
