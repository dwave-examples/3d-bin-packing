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
from dash import ALL, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from demo_configs import RANDOM_SEED, TABLE_HEADERS
import numpy as np
import plotly.graph_objs as go

from demo_interface import generate_problem_details_table_rows, generate_table
from packing3d import Bins, Cases, Variables, build_cqm, call_solver
from src.demo_enums import ProblemType, SolverType
from utils import case_list_to_dict, data_to_lists, get_cqm_stats, plot_cuboids, update_colors, write_input_data, write_solution_to_file


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
        return ["display-none"]*len(gen_settings), ""

    return [""]*len(gen_settings), "display-none"


@dash.callback(
    Output("input", "children"),
    Output("data-table-store", "data"),
    Output("saved", "className", allow_duplicate=True),
    inputs=[
        Input("problem-type", "value"),
        Input("num-cases", "value"),
        Input("case-dim", "value"),
    ],
    prevent_initial_call='initial_duplicate'
)
def update_input_table_generated(
    problem_type: Union[ProblemType, int],
    num_cases: int,
    case_size_range: list[int],
) -> tuple[list, list, str]:
    """Updates the input table when ProblemType is `Generated` and any relevant settings have been
    changed.

    Args:
        problem_type: The input problem type. Either Generated or Uploaded.
        num_cases: The value of the number of cases setting.
        case_size_range: The values of the case size range setting.

    Returns:
        list: The input table.
        list: The data that was generated for the table.
        str: The class name for the `Saved!` feedback.
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

    return generate_table(TABLE_HEADERS, data_lists), data_lists, "display-none"


@dash.callback(
    Output("max-bins", "children"),
    Output("bin-dims", "children"),
    Output("max-bins-store", "data"),
    Output("bin-dimensions-store", "data"),
    Output("saved", "className"),
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
) -> tuple[int, str, int, list, str]:
    """Updates the number of bins and bin dimensions store and input when ProblemType is `Generated`
    and any relevant form fields are updated.

    Args:
        problem_type: The input problem type. Either Generated or Uploaded.
        num_bins: The current value of the number of bins setting.
        bin_length: The current value of the bin length setting.
        bin_width: The current value of the bin width setting.
        bin_height: The current value of the bin height setting.

    Returns:
        max_bins: The maximum bins to display in the input UI.
        bin_dimensions: The bin dimension string to display in the UI.
        max_bins_store: The value to update the maximum bins store.
        bin_dimensions_store: The value to update the bin dimensions store.
        saved_classname: The `Saved!` text class name.
    """
    if ProblemType(problem_type) is ProblemType.FILE:
        raise PreventUpdate

    return (
        num_bins,
        f"{bin_length} * {bin_width} * {bin_height}",
        num_bins,
        [bin_length, bin_width, bin_height],
        "display-none"
    )


class UpdateInputFileReturn(NamedTuple):
    """Return type for the ``update_input_file`` callback function."""

    table_input: list = dash.no_update
    max_bins: int = dash.no_update
    bin_dimensions: str = dash.no_update
    filename: str = dash.no_update
    data_table_store: list = dash.no_update
    max_bins_store: int = dash.no_update
    bin_dimensions_store: list = dash.no_update

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
) -> UpdateInputFileReturn:
    """Reads input file and displays data in a table.

    Args:
        file_contents: The encoded contents of the uploaded input file.
        problem_type: The input problem type. Either Generated or Uploaded.
        filename: The name of the uploaded file.

    Returns:
        A NamedTuple (UpdateInputFileReturn) with the following parameters:
            table_input: The input table containing problem data from the file.
            max_bins: The maximum bins to display in the input UI.
            bin_dimensions: The bin dimension string to display in the UI.
            filename: The name of the file that was uploaded to display in the UI.
            data_table_store: The value to update the table data store.
            max_bins_store: The value to update the maximum bins store.
            bin_dimensions_store: The value to update the bin dimensions store.
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
            return UpdateInputFileReturn(filename='There was an error processing this file.')

        return UpdateInputFileReturn(
            table_input=generate_table(TABLE_HEADERS, table_data),
            max_bins=num_bins,
            bin_dimensions=f"{bin_length} * {bin_width} * {bin_height}",
            filename=filename,
            data_table_store=table_data,
            max_bins_store=num_bins,
            bin_dimensions_store=[bin_length, bin_width, bin_height],
        )

    raise PreventUpdate


@dash.callback(
    Output("saved", "className", allow_duplicate=True),
    inputs=[
        Input("save-input-button", "n_clicks"),
        State("save-input-filename", "value"),
        State("data-table-store", "data"),
        State("max-bins-store", "data"),
        State("bin-dimensions-store", "data"),
    ],
    prevent_initial_call=True,
)
def save_input_to_file(
    save_button: int,
    filename: str,
    data_table: list[int],
    num_bins: int,
    bin_dimensions: list[int],
) -> str:
    """Saves input data to a text file when the `save-input-button` is clicked.

    Args:
        save_button: How many times the save to file button has been clicked.
        filename: The file name to save the input data to.
        data_table: The data from the table of input values.
        num_bins: The number of bins.
        bin_dimensions: The bin dimensions.

    Returns:
        str: The `Saved!` text class name.
    """
    write_input_data(case_list_to_dict(num_bins, bin_dimensions, data_table), filename)
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
        State("data-table-store", "data"),
        State("max-bins-store", "data"),
        State("bin-dimensions-store", "data"),
        State("checklist", "value"),
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
        data_table: The stored generated data.
        num_bins: The stored number of bins.
        bin_dimensions: The stored bin dimensions.
        checklist: The current value of the checklist.
        save_solution_filepath: The filepath to save the solution to.

    Returns:
        fig: The results figure.
        problem_details_table: The table and information to display in the problem details table.
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
                       bins, effective_dimensions, bool(checklist))

     # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(get_cqm_stats(cqm))

    return fig, problem_details_table
