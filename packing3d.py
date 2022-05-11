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

import argparse
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet
from dwave.system import LeapHybridCQMSampler
from itertools import combinations, permutations
import numpy as np
from typing import Tuple

from utils import print_cqm_stats, plot_cuboids
from utils import read_instance, write_solution_to_file
from mip_solver import MIPCQMSolver


class Cases:
    """Class for representing cuboid item data in a 3D bin packing problem.

    Args:
         data: dictionary containing raw information for both bins and cases
    
    """

    def __init__(self, data):
        self.case_ids = np.repeat(data["case_ids"], data["quantity"])
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)
        self.length = np.repeat(data["case_length"], data["quantity"])
        self.width = np.repeat(data["case_width"], data["quantity"])
        self.height = np.repeat(data["case_height"], data["quantity"])
        print(f'Number of cases: {self.num_cases}')


class Bins:
    """Class for representing cuboid container data in a 3D bin packing problem.

    Args:
        data: dictionary containing raw information for both bins and cases
        cases: Instance of ``Cases``, representing cuboid items packed into containers.

    """

    def __init__(self, data, cases):
        self.length = data["bin_dimensions"][0]
        self.width = data["bin_dimensions"][1]
        self.height = data["bin_dimensions"][2]
        self.num_bins = data["num_bins"]
        self.lowest_num_bin = np.ceil(
            np.sum(cases.length * cases.width * cases.height) / (
                    self.length * self.width * self.height))
        if self.lowest_num_bin > self.num_bins:
            raise RuntimeError(
                f'number of bins is at least {self.lowest_num_bin}, ' +
                'try increasing the number of bins'
            )
        print(f'Minimum Number of bins required: {self.lowest_num_bin}')


class Variables:
    """Class that collects all CQM model variables for the 3D bin packing problem.

    Args:
        cases: Instance of ``Cases``, representing cuboid items packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
    
    """

    def __init__(self, cases: Cases, bins: Bins):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        self.x = {i: Real(f'x_{i}',
                          lower_bound=0,
                          upper_bound=bins.length * bins.num_bins)
                  for i in range(num_cases)}
        self.y = {i: Real(f'y_{i}', lower_bound=0, upper_bound=bins.width)
                  for i in range(num_cases)}
        self.z = {i: Real(f'z_{i}', lower_bound=0, upper_bound=bins.height)
                  for i in range(num_cases)}

        self.bin_height = {
            j: Real(label=f'upper_bound_{j}', upper_bound=bins.height)
            for j in range(num_bins)}

        self.bin_loc = {
            (i, j): Binary(f'case_{i}_in_bin_{j}') if num_bins > 1 else 1
            for i in range(num_cases) for j in range(num_bins)}

        self.bin_on = {j: Binary(f'bin_{j}_is_used') if num_bins > 1 else 1
                       for j in range(num_bins)}

        self.o = {(i, k): Binary(f'o_{i}_{k}') for i in range(num_cases)
                  for k in range(6)}

        self.selector = {(i, j, k): Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}


def _add_bin_on_constraint(cqm: ConstrainedQuadraticModel, vars: Variables,
                           bins: Bins, cases: Cases):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    if num_bins > 1:
        for j in range(num_bins):
            cqm.add_constraint((1 - vars.bin_on[j]) * quicksum(
                vars.bin_loc[i, j] for i in range(num_cases)) <= 0,
                               label=f'bin_on_{j}')

        for j in range(num_bins - 1):
            cqm.add_constraint(vars.bin_on[j] - vars.bin_on[j + 1] >= 0,
                               label=f'bin_use_order_{j}')


def _add_orientation_constraints(cqm: ConstrainedQuadraticModel,
                                 vars: Variables, cases: Cases) -> list:
    num_cases = cases.num_cases
    dx = {}
    dy = {}
    dz = {}
    for i in range(num_cases):
        p1 = list(
            permutations([cases.length[i], cases.width[i], cases.height[i]]))
        dx[i] = 0
        dy[i] = 0
        dz[i] = 0
        for j, (a, b, c) in enumerate(p1):
            dx[i] += a * vars.o[i, j]
            dy[i] += b * vars.o[i, j]
            dz[i] += c * vars.o[i, j]

    for i in range(num_cases):
        cqm.add_discrete(quicksum([vars.o[i, k] for k in range(6)]),
                         label=f'orientation_{i}')
    return [dx, dy, dz]


def _add_geometric_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                               bins: Bins, cases: Cases,
                               effective_dimensions: list):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    dx, dy, dz = effective_dimensions

    for i, k in combinations(range(num_cases), r=2):
        cqm.add_discrete(quicksum([vars.selector[i, k, s] for s in range(6)]),
                         label=f'discrete_{i}_{k}')
        for j in range(num_bins):
            cases_on_same_bin = vars.bin_loc[i, j] * vars.bin_loc[k, j]
            cqm.add_constraint(
                - (2 - cases_on_same_bin -
                   vars.selector[i, k, 0]) * num_bins * bins.length +
                (vars.x[i] + dx[i] - vars.x[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_0')

            cqm.add_constraint(
                -(2 - cases_on_same_bin -
                  vars.selector[i, k, 1]) * bins.width +
                (vars.y[i] + dy[i] - vars.y[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_1')

            cqm.add_constraint(
                -(2 - cases_on_same_bin -
                  vars.selector[i, k, 2]) * bins.height +
                (vars.z[i] + dz[i] - vars.z[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_2')

            cqm.add_constraint(
                -(2 - cases_on_same_bin -
                  vars.selector[i, k, 3]) * num_bins * bins.length +
                (vars.x[k] + dx[k] - vars.x[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_3')

            cqm.add_constraint(
                -(2 - cases_on_same_bin -
                  vars.selector[i, k, 4]) * bins.width +
                (vars.y[k] + dy[k] - vars.y[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_4')

            cqm.add_constraint(
                -(2 - cases_on_same_bin -
                  vars.selector[i, k, 5]) * bins.height +
                (vars.z[k] + dz[k] - vars.z[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_5')

    if num_bins > 1:
        for i in range(num_cases):
                cqm.add_discrete(
                quicksum([vars.bin_loc[i, j] for j in range(num_bins)]),
                label=f'case_{i}_max_packed')


def _add_boundary_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                              bins: Bins, cases: Cases,
                              effective_dimensions: list):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    dx, dy, dz = effective_dimensions
    for i in range(num_cases):
        for j in range(num_bins):
            cqm.add_constraint(vars.z[i] + dz[i] - vars.bin_height[j] -
                               (1 - vars.bin_loc[i, j]) * bins.height <= 0,
                               label=f'maxx_height_{i}_{j}')

            cqm.add_constraint(vars.x[i] + dx[i] - bins.length * (j + 1)
                               - (1 - vars.bin_loc[i, j]) *
                               num_bins * bins.length <= 0,
                               label=f'maxx_{i}_{j}_less')

            cqm.add_constraint(
                vars.x[i] - bins.length * j * vars.bin_loc[i, j] >= 0,
                label=f'maxx_{i}_{j}_greater')

            cqm.add_constraint(
                vars.y[i] + dy[i] <= bins.width,
                label=f'maxy_{i}_{j}_less')


def _define_objective(cqm: ConstrainedQuadraticModel, vars: Variables,
                      bins: Bins, cases: Cases, effective_dimensions: list):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    dx, dy, dz = effective_dimensions

    # First term of objective: minimize average height of cases
    first_obj_term = quicksum(
        vars.z[i] + dz[i] for i in range(num_cases)) / num_cases

    # Second term of objective: minimize height of the case at the top of the
    # bin
    second_obj_term = quicksum(vars.bin_height[j] for j in range(num_bins))

    # Third term of the objective:
    third_obj_term = quicksum(
        bins.height * vars.bin_on[j] for j in range(num_bins))
    first_obj_coefficient = 1
    second_obj_coefficient = 1
    third_obj_coefficient = 1
    cqm.set_objective(first_obj_coefficient * first_obj_term +
                      second_obj_coefficient * second_obj_term +
                      third_obj_coefficient * third_obj_term)


def build_cqm(vars: Variables, bins: Bins,
              cases: Cases) -> Tuple[ConstrainedQuadraticModel, list]:
    """Builds the CQM model from the problem variables and data.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        cases: Instance of ``Cases``, representing cuboid items packed into containers.

    Returns:
        A ``dimod.CQM`` object that defines the 3D bin packing problem.
        effective_dimensions: List of case dimensions based on orientations of cases.
    
    """
    cqm = ConstrainedQuadraticModel()
    effective_dimensions = _add_orientation_constraints(cqm, vars, cases)
    _add_bin_on_constraint(cqm, vars, bins, cases)
    _add_geometric_constraints(cqm, vars, bins, cases, effective_dimensions)
    _add_boundary_constraints(cqm, vars, bins, cases, effective_dimensions)
    _define_objective(cqm, vars, bins, cases, effective_dimensions)

    return cqm, effective_dimensions


def call_solver(cqm: ConstrainedQuadraticModel,
                time_limit: float,
                use_cqm_solver: bool = True) -> SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``CQM`` object that defines the 3D bin packing problem.
        time_limit: Time limit parameter to pass on to the CQM sampler.

    Returns:
        A ``dimod.SampleSet`` that represents the best feasible solution found.
    
    """
    if use_cqm_solver:
        sampler = LeapHybridCQMSampler()
        res = sampler.sample_cqm(cqm, time_limit=time_limit, label='3d bin packing')
    else:
        sampler = MIPCQMSolver()
        res = sampler.sample_cqm(cqm, time_limit=time_limit)

    res.resolve()
    feasible_sampleset = res.filter(lambda d: d.is_feasible)
    print(feasible_sampleset)
    try:
        best_feasible = feasible_sampleset.first.sample

        return best_feasible
        
    except ValueError:
        raise RuntimeError(
            "Sampleset is empty, try increasing time limit or " +
            "adjusting problem config."
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, nargs="?",
                        help="Filename with path to bin-packing data file.",
                        default="input/sample_data_1.txt")
    
    parser.add_argument("--output_filepath", type=str,  nargs="?",
                        help="Path for the output solution file.",
                        default=None)

    parser.add_argument("--time_limit", type=float, nargs="?",
                        help="Time limit for the hybrid CQM Solver to run in"
                             " seconds.",
                        default=20)
    
    parser.add_argument("--use_cqm_solver", type=bool, nargs="?",
                        help="Flag to either use CQM or MIP solver",
                        default=True)
    
    parser.add_argument("--html_filepath", type=str, nargs="?",
                        help="Filename with path to plot html file.",
                        default=None)

    parser.add_argument("--color_coded", type=bool, nargs="?",
                        help="View plot with coded or randomly colored cases.",
                        default=False)

    args = parser.parse_args()
    output_filepath = args.output_filepath
    time_limit = args.time_limit
    use_cqm_solver = args.use_cqm_solver
    html_filepath = args.html_filepath
    color_coded = args.color_coded

    data = read_instance(args.data_filepath)
    cases = Cases(data)
    bins = Bins(data, cases)

    vars = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(vars, bins, cases)

    print_cqm_stats(cqm)

    best_feasible = call_solver(cqm, time_limit, use_cqm_solver)

    if output_filepath is not None:
        write_solution_to_file(output_filepath, cqm, vars, best_feasible, 
                               cases, bins, effective_dimensions)

    fig = plot_cuboids(best_feasible, vars, cases,
                       bins, effective_dimensions, color_coded)

    if html_filepath is not None:
        fig.write_html(html_filepath)

    fig.show()
