# Copyright 2024 D-Wave
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

import os
from demo_configs import TABLE_HEADERS
import plotly.colors as colors
import plotly.graph_objects as go
import numpy as np
from tabulate import tabulate
from typing import List, Optional, TYPE_CHECKING
import dimod

if TYPE_CHECKING:
    from packing3d import Cases, Bins, Variables


def get_cqm_stats(cqm: dimod.ConstrainedQuadraticModel) -> list[list]:
    """Get information about the CQM model defining the 3D bin packing problem.

    Args:
        cqm: A dimod constrained quadratic model.

    Returns:
        list[list]: A list of CQM data ready to be printed or displayed in a table.
    """
    if not isinstance(cqm, dimod.ConstrainedQuadraticModel):
        raise ValueError("input instance should be a dimod CQM model")
    num_binaries = sum(cqm.vartype(v) is dimod.BINARY for v in cqm.variables)
    num_integers = sum(cqm.vartype(v) is dimod.INTEGER for v in cqm.variables)
    num_continuous = sum(cqm.vartype(v) is dimod.REAL for v in cqm.variables)
    num_discretes = len(cqm.discrete)
    num_linear_constraints = sum(
        constraint.lhs.is_linear() for constraint in cqm.constraints.values())
    num_quadratic_constraints = sum(
        not constraint.lhs.is_linear() for constraint in
        cqm.constraints.values())
    num_le_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Le for constraint in
        cqm.constraints.values())
    num_ge_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Ge for constraint in
        cqm.constraints.values())
    num_equality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Eq for constraint in
        cqm.constraints.values())

    assert (num_binaries + num_integers + num_continuous == len(cqm.variables))

    assert (num_quadratic_constraints + num_linear_constraints ==
            len(cqm.constraints))

    return [
        ["Binary", "Integer", "Continuous", "Quad", "Linear", "One-hot", "EQ", "LT", "GT"],
        [
            num_binaries,
            num_integers,
            num_continuous,
            num_quadratic_constraints,
            num_linear_constraints,
            num_discretes,
            num_equality_constraints,
            num_le_inequality_constraints,
            num_ge_inequality_constraints
        ]
    ]


def print_cqm_stats(cqm: dimod.ConstrainedQuadraticModel) -> None:
    """Print some information about the CQM model defining the 3D bin packing problem.

    Args:
        cqm: A dimod constrained quadratic model.

    """
    cqm_stats = get_cqm_stats(cqm)

    print(" \n" + "=" * 35 + "MODEL INFORMATION" + "=" * 35)
    print(
        ' ' * 10 + 'Variables' + " " * 20 + 'Constraints' + " " * 15 +
        'Sensitivity')
    print('-' * 30 + " " + '-' * 28 + ' ' + '-' * 18)
    print(tabulate(cqm_stats, headers="firstrow"))


def update_colors(fig: go.Figure, color_coded: bool) -> go.Figure:
    """Update the case colors given a figure and the color coding rule.

    Args:
        fig: A go.Figure containing colored cases with case IDs of the form ``case_#``.
        color_coded: Whether the cases should be colored based on their ID.

    Returns:
        go.Figure: The go.Figure with updated colors.
    """
    if color_coded:
        case_ids = np.array(
            [
                int(trace["name"].split("_")[1]) for trace in fig["data"] if trace["name"][:4] == "case"
            ]
        )
        colors = _get_colors(case_ids)

    for i, trace in enumerate(fig["data"]):
        if trace["name"][:4] == "case":
            trace["color"] = colors[i] if color_coded else ""

    return fig


def _cuboid_data(origin: tuple, size: tuple = (1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(origin)

    return X


def _get_all_cuboids(positions: List[tuple], sizes: List[tuple],
                     color_coded: bool, case_ids: np.array) -> list:
    case_data = []
    mesh_kwargs = dict(alphahull=0, flatshading=True, showlegend=True)
    colors = _get_colors(case_ids)
    for p, s, c, id in zip(positions, sizes, colors, case_ids):
        case_points = _cuboid_data(p, size=s)
        # Get all unique vertices for 3d Mesh
        x, y, z = np.unique(np.vstack(case_points), axis=0).T
        if color_coded:
            mesh_kwargs["color"] = c
        case_data.append(go.Mesh3d(x=x, y=y, z=z,
                                   name=f"case_{id}",
                                   **mesh_kwargs))

    return case_data


def _plot_cuboids(positions: List[tuple], sizes: List[tuple],
                  bin_length: int, bin_width: int,
                  bin_height: int, color_coded: bool,
                  case_ids: np.array) -> go.Figure:
    case_data = _get_all_cuboids(positions, sizes, color_coded, case_ids)
    fig = go.Figure(data=case_data)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0, bin_length * 1.1]),
        yaxis=dict(range=[0, bin_width * 1.1]),
        zaxis=dict(range=[0, bin_height * 1.1])
    ))

    return fig


def _get_colors(case_ids: np.array) -> list:
    if len(np.unique(case_ids)) > 1:
        scaled = (case_ids - np.min(case_ids)) / \
                 (np.max(case_ids) - np.min(case_ids))
        return colors.sample_colorscale(colors.sequential.Rainbow, scaled)

    return ["blue"] * len(case_ids)


def plot_cuboids(sample: dimod.SampleSet, vars: "Variables",
                 cases: "Cases", bins: "Bins", effective_dimensions: list,
                 color_coded: bool = True) -> go.Figure:
    """Visualization utility tool to view 3D bin packing solution.

    Args:
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        cases: Instance of ``Cases``, representing cuboid items packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        effective_dimensions: List of case dimensions based on orientations of cases.

    Returns:
        ``plotly.graph_objects.Figure`` with all cases packed according to CQM results.

    """
    dx, dy, dz = effective_dimensions
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    positions = []
    sizes = []
    for i in range(num_cases):
        positions.append(
            (vars.x[i].energy(sample), vars.y[i].energy(sample),
             vars.z[i].energy(sample)))
        sizes.append((dx[i].energy(sample),
                      dy[i].energy(sample),
                      dz[i].energy(sample)))
    fig = _plot_cuboids(positions, sizes, bins.length * num_bins,
                        bins.width, bins.height, color_coded, cases.case_ids)
    for i in range(num_bins):
        fig.add_trace(
            go.Scatter3d(x=[bins.length * i, bins.length * (i + 1)], y=[0, 0],
                         z=[0, 0], mode='lines', name=f"Bin Boundary {i + 1}",
                         line_color="red", line_width=5)
        )
        fig.add_trace(
            go.Scatter3d(x=[bins.length * (i + 1)] * 2, y=[0, bins.width],
                         z=[0, 0], mode='lines', name=f"Bin Boundary {i + 1}",
                         line_color="red", line_width=5)
        )
        fig.add_trace(
            go.Scatter3d(x=[bins.length * (i + 1)] * 2, y=[0, 0],
                         z=[0, bins.height], mode='lines',
                         name=f"Bin Boundary {i + 1}", line_color="red",
                         line_width=5)
        )

    fig.update_layout(scene=dict(aspectmode="data"))

    return fig


def case_list_to_dict(num_bins: int, bin_dimensions: list, case_info: list) -> dict:
    """Convert instance input files into raw problem data.

    Args:
        instance_path:  Path to the bin packing problem instance file.

    Returns:
        data: dictionary containing raw information for both bins and cases.
    """

    data = {"num_bins": num_bins, "bin_dimensions": bin_dimensions, "quantity": [], "case_ids": [],
            "case_length": [], "case_width": [], "case_height": []}

    for case_info_line in case_info:
        data["case_ids"].append(case_info_line[0])
        data["quantity"].append(case_info_line[1])
        data["case_length"].append(case_info_line[2])
        data["case_width"].append(case_info_line[3])
        data["case_height"].append(case_info_line[4])

    return data


def read_instance(instance_path: str) -> dict:
    """Convert instance input files into raw problem data.

    Args:
        instance_path:  Path to the bin packing problem instance file.

    Returns:
        data: dictionary containing raw information for both bins and cases.

    """
    with open(instance_path) as f:
        case_info = []

        for i, line in enumerate(f):
            if i == 0:
                num_bins = int(line.split()[-1])
            elif i == 1:
                bin_dimensions = [int(i) for i in line.split()[-3:]]
            elif 2 <= i <= 4:
                continue
            else:
                case_info.append(list(map(int, line.split())))

        return case_list_to_dict(num_bins, bin_dimensions, case_info)


def write_solution_to_file(solution_file_path: str,
                           cqm: dimod.ConstrainedQuadraticModel,
                           vars: "Variables",
                           sample: dimod.SampleSet,
                           cases: "Cases",
                           bins: "Bins",
                           effective_dimensions: list):
    """Write solution to a file.

    Args:
        solution_file_path: path to the output solution file. If doesn't exist,
            a new file is created.
        cqm: A ``dimod.CQM`` object that defines the 3D bin packing problem.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        cases: Instance of ``Cases``, representing cases packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        effective_dimensions: List of case dimensions based on orientations of cases.

    """
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    lowest_num_bin = bins.lowest_num_bin
    dx, dy, dz = effective_dimensions
    if num_bins > 1:
        num_bin_used = lowest_num_bin + sum([vars.bin_on[j].energy(sample)
                            for j in range(lowest_num_bin, num_bins)])
    else:
        num_bin_used = 1

    objective_value = cqm.objective.energy(sample)
    vs = [['case_id', 'bin-location', 'orientation', 'x', 'y', 'z', "x'",
           "y'", "z'"]]
    for i in range(num_cases):
        vs.append([cases.case_ids[i],
                   int(sum(int(j == 0) if i == 0 or num_bins == 1 else
                           (j + 1) * vars.bin_loc[i, j].energy(sample)
                           for j in range(num_bins))),
                   int(sum((r + 1) * vars.o[i, r].energy(sample) for r in
                           range(6))),
                   np.round(vars.x[i].energy(sample), 2),
                   np.round(vars.y[i].energy(sample), 2),
                   np.round(vars.z[i].energy(sample), 2),
                   np.round(dx[i].energy(sample), 2),
                   np.round(dy[i].energy(sample), 2),
                   np.round(dz[i].energy(sample), 2)])

    with open(solution_file_path, 'w') as f:
        f.write('# Number of bins used: ' + str(int(num_bin_used)) + '\n')
        f.write('# Number of cases packed: ' + str(int(num_cases)) + '\n')
        f.write(
            '# Objective value: ' + str(np.round(objective_value, 3)) + '\n\n')
        f.write(tabulate(vs, headers="firstrow"))
        f.close()
        print(f'Saved solution to '
              f'{os.path.join(os.getcwd(), solution_file_path)}')


def write_input_data(data: dict, input_filename: Optional[str] = None) -> str:
    """Convert input data dictionary to an input string and write it to a file.

    Args:
        data: dictionary containing raw information for both bins and cases
        input_filename: name of the file for writing input data

    Returns:
        input_string: input data information

    """
    case_info = data_to_lists(data)

    input_string = f'# Max num of bins : {data["num_bins"]} \n'
    input_string += (f'# Bin dimensions '
                     f'(L * W * H): {data["bin_dimensions"][0]} '
                     f'{data["bin_dimensions"][1]} '
                     f'{data["bin_dimensions"][2]} \n \n')
    input_string += tabulate([TABLE_HEADERS, *[v for v in case_info]],
                             headers="firstrow", colalign='right')

    if input_filename is not None:
        full_file_path = os.path.join("input", input_filename)
        f = open(full_file_path, "w")
        f.write(input_string)
        f.close()

    return input_string


def data_to_lists(data: dict) -> list[int]:
    """Convert input data dictionary to a list to prepare for display.

    Args:
        data: dictionary containing raw information for both bins and cases

    Returns:
        case_info: a list of lists of rows that fall under the headers.
    """
    case_info = [[i, data["quantity"][i], data["case_length"][i],
                  data["case_width"][i], data["case_height"][i]]
                 for i in range(len(data['case_ids']))]

    return case_info
