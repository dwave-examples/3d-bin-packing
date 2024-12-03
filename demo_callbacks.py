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
from typing import NamedTuple, Union

import dash
from dash import ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from demo_configs import RANDOM_SEED, TABLE_HEADERS
import numpy as np
import plotly.graph_objs as go

from demo_interface import generate_table
from packing3d import Bins, Cases, Variables, build_cqm, call_solver
from src.demo_enums import ProblemType, SolverType
from utils import case_list_to_dict, data_to_lists, plot_cuboids, write_solution_to_file


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
    inputs=[
        Input("problem-type", "value"),
        State({"type": "generated-settings", "index": ALL}, "children"),
    ],
)
def update_problem_type(problem_type: Union[ProblemType, int], gen_settings: list) -> tuple[str, str]:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    if problem_type is ProblemType.FILE.value:
        return ["display-none"]*len(gen_settings), ""

    return [""]*len(gen_settings), "display-none"


@dash.callback(
    Output("input", "children"),
    Output("data-table-store", "data"),
    inputs=[
        Input("problem-type", "value"),
        Input("num-cases", "value"),
        Input("case-dim", "value"),
    ],
)
def update_input_graph_generated(
    problem_type: Union[ProblemType, int],
    num_cases: int,
    case_size_range: int,
) -> str:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    if ProblemType(problem_type) is ProblemType.FILE:
        raise PreventUpdate

    rng = np.random.default_rng(RANDOM_SEED)

    data = {
        "case_length": rng.integers(
            case_size_range[0], case_size_range[1], 
            num_cases, endpoint=True
        ),
        "case_width": rng.integers(
            case_size_range[0], case_size_range[1], 
            num_cases, endpoint=True
        ),
        "case_height": rng.integers(
            case_size_range[0], case_size_range[1], 
            num_cases, endpoint=True
        ),
    }

    # Determine quantities and case_ids
    case_dimensions = np.vstack(
        [data["case_length"], data["case_width"], data["case_height"]]
    )
    unique_dimensions, data["quantity"] = np.unique(case_dimensions, 
                                                    axis=1,
                                                    return_counts=True)
    
    data["case_length"] = unique_dimensions[0,:]
    data["case_width"] = unique_dimensions[1,:]
    data["case_height"] = unique_dimensions[2,:]
    
    data["case_ids"] = np.array(range(len(data["quantity"])))

    data_lists = data_to_lists(data)

    return generate_table(TABLE_HEADERS, data_lists), data_lists


@dash.callback(
    Output("max-bins", "children"),
    Output("bin-dims", "children"),
    Output("max-bins-store", "data"),
    Output("bin-dimensions-store", "data"),
    inputs=[
        Input("problem-type", "value"),
        Input("num-bins", "value"),
        Input("bin-length", "value"),
        Input("bin-width", "value"),
        Input("bin-height", "value"),
    ],
)
def update_input_generated(
    problem_type: Union[ProblemType, int],
    num_bins: int,
    bin_length: int,
    bin_width: int,
    bin_height: int,
) -> str:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    if ProblemType(problem_type) is ProblemType.FILE:
        raise PreventUpdate

    return num_bins, f"{bin_length} * {bin_width} * {bin_height}", num_bins, [bin_length, bin_width, bin_height]


@dash.callback(
    Output("input", "children", allow_duplicate=True),
    Output("max-bins", "children", allow_duplicate=True),
    Output("bin-dims", "children", allow_duplicate=True),
    Output("filename", "children"),
    Output("data-table-store", "data", allow_duplicate=True),
    Output("max-bins-store", "data", allow_duplicate=True),
    Output("bin-dimensions-store", "data", allow_duplicate=True),
    inputs=[
        Input("input-file", 'contents'),
        Input("problem-type", "value"),
        State("input-file", 'filename'),
    ],
    prevent_initial_call=True,
)
def update_input_file(
    file_contents: str,
    problem_type: Union[ProblemType, int],
    filename: str,
) -> str:
    """Runs on load and any time the value of the slider is updated.
        Add `prevent_initial_call=True` to skip on load runs.

    Args:
        slider_value: The value of the slider.

    Returns:
        str: The content of the input tab.
    """
    if ProblemType(problem_type) is ProblemType.GENERATED:
        raise PreventUpdate

    if file_contents is not None:
        decoded = base64.b64decode(file_contents)

        try:
            lines = decoded.decode('ISO-8859-1').splitlines()

            num_bins = int(lines[0].split(":")[1].strip())
            bin_length, bin_width, bin_height = map(int, lines[1].split(":")[1].split())

            table_data = []
            for line in lines[5:]:
                if line.strip():
                    table_data.append(list(map(int, line.split())))

        except Exception as e:
            print(e)
            return 'There was an error processing this file.'

        return (
            generate_table(TABLE_HEADERS, table_data),
            num_bins,
            f"{bin_length} * {bin_width} * {bin_height}",
            filename,
            table_data,
            num_bins,
            [bin_length, bin_width, bin_height]
        )

    raise PreventUpdate


@dash.callback(
    # The Outputs below must align with `RunOptimizationReturn`.
    Output("results", "figure"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-type-select", "value"),
        State("solver-time-limit", "value"),
        State("data-table-store", "data"),
        State("max-bins-store", "data"),
        State("bin-dimensions-store", "data"),
        State("checklist", "value"),
        State("save-input", "value"),
        State("save-solution", "value"),
    ],
    running=[
        (Output("cancel-button", "className"), "", "display-none"),  # Show/hide cancel button.
        (Output("run-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    data_table: list[int],
    num_bins: int,
    bin_dimensions: list[int],
    checklist: list,
    save_input: str,
    save_solution_filepath: str,
) -> go.Figure:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the ``Run Optimization`` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and updates all relevant HTML
    components.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        solver_type: The solver to use for the optimization run defined by SolverType in demo_enums.py.
        time_limit: The solver time limit.

    Returns:
        A NamedTuple (RunOptimizationReturn) containing all outputs to be used when updating the HTML
        template (in ``demo_interface.py``). These are:

            results: The results to display in the results tab.
            problem-details: List of the table rows for the problem details table.
    """
    solver_type = SolverType(solver_type)

    data = case_list_to_dict(num_bins, bin_dimensions, data_table)
    cases = Cases(data)
    bins = Bins(data, cases)

    vars = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(vars, bins, cases)

    best_feasible = call_solver(cqm, time_limit, solver_type is SolverType.CQM)

    if save_solution_filepath is not None:
        write_solution_to_file(save_solution_filepath, cqm, vars, best_feasible,
                               cases, bins, effective_dimensions)

    fig = plot_cuboids(best_feasible, vars, cases,
                       bins, effective_dimensions, checklist)

    return fig
