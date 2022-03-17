import os
import time
from typing import Tuple

import argparse
from itertools import combinations, permutations
import numpy as np
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet

from utils import print_cqm_stats, plot_cuboids

# todo: remove this before launch
use_local = True
if use_local:
    from cqmsolver.hss_sampler import HSSCQMSampler
else:
    from dwave.system import LeapHybridSampler


class Cases:
    """Class for representing cuboid item data in a 3D bin packing problem.

    Args:
        data_filepath: Specifies file to load comma-delimited data from.
            See sample_data.txt for format.
    
    """
    def __init__(self, data_filepath: str):
        path = os.path.join(os.path.dirname(__file__), data_filepath)
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int32)
        self.case_ids = data["case_id"]
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)
        self.length = np.repeat(data["length"], data["quantity"])
        self.width = np.repeat(data["width"], data["quantity"])
        self.height = np.repeat(data["height"], data["quantity"])
        print(f'Number of cases: {self.num_cases}')


class Pallets:
    """Class for representing cuboid container data in a 3D bin packing problem.

    Args:
        length: Length of container(s).
        width: Width of container(s).
        height: Height of container(s).
        num_pallets: Number of containers.
        cases: Instance of ``Cases``, representing items packed into containers.

    """

    def __init__(self, length: int, width: int, height: int, num_pallets: int,
                 cases: Cases):
        self.length = length
        self.width = width
        self.height = height
        self.num_pallets = num_pallets
        self.lowest_num_pallet = np.ceil(
            np.sum(cases.length * cases.width * cases.height) / (
                    length * width * height))
        assert self.lowest_num_pallet <= num_pallets, \
            f'number of pallets is at least {self.lowest_num_pallet}'
        print(f'Minimum Number of pallets required: {self.lowest_num_pallet}')


class Variables:
    """Class that collects all CQM model variables for the 3D bin packing problem.

    Args:
        cases: Instance of ``Cases``, representing items packed into containers.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
    
    """
    def __init__(self, cases: Cases, pallets: Pallets):
        num_cases = cases.num_cases
        num_pallets = pallets.num_pallets
        self.x = {i: Real(f'x_{i}',
                          lower_bound=0,
                          upper_bound=pallets.length * pallets.num_pallets)
                  for i in range(num_cases)}
        self.y = {i: Real(f'y_{i}', lower_bound=0, upper_bound=pallets.width)
                  for i in range(num_cases)}
        self.z = {i: Real(f'z_{i}', lower_bound=0, upper_bound=pallets.height)
                  for i in range(num_cases)}

        self.bin_height = {
            j: Real(label=f'upper_bound_{j}', upper_bound=pallets.height)
            for j in range(num_pallets)}

        self.pallet_loc = {(i, j): Binary(f'box_{i}_in_bin_{j}')
                           for i in range(num_cases)
                           for j in range(num_pallets)}

        self.pallet_on = {j: Binary(f'bin_{j}_is_used')
                          for j in range(num_pallets)}

        self.o = {(i, k): Binary(f'o_{i}_{k}') for i in range(num_cases)
                  for k in range(6)}

        self.selector = {(i, j, k): Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}


def _add_pallet_on_constraint(cqm: ConstrainedQuadraticModel, vars: Variables,
                              pallets: Pallets, cases: Cases):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    for j in range(num_pallets):
        cqm.add_constraint((1 - vars.pallet_on[j]) * quicksum(
            vars.pallet_loc[i, j] for i in range(num_cases)) <= 0,
                           label=f'p_on_{j}')

    for j in range(num_pallets - 1):
        cqm.add_constraint(vars.pallet_on[j] - vars.pallet_on[j + 1] >= 0,
                           label=f'bin_use_order_{j}')


def _add_orientation_constraints(cqm: ConstrainedQuadraticModel,
                                 vars: Variables, cases: Cases) -> list:
    num_cases = cases.num_cases
    ox = {}
    oy = {}
    oz = {}
    for i in range(num_cases):
        p1 = list(
            permutations([cases.length[i], cases.width[i], cases.height[i]]))
        ox[i] = 0
        oy[i] = 0
        oz[i] = 0
        for j, (a, b, c) in enumerate(p1):
            ox[i] += a * vars.o[i, j]
            oy[i] += b * vars.o[i, j]
            oz[i] += c * vars.o[i, j]

    for i in range(num_cases):
        cqm.add_discrete(quicksum([vars.o[i, k] for k in range(6)]),
                         label=f'orientation_{i}')
    return [ox, oy, oz]


def _add_geometric_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                               pallets: Pallets, cases: Cases, origins: list):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins

    for i, k in combinations(range(num_cases), r=2):
        cqm.add_discrete(quicksum([vars.selector[i, k, s] for s in range(6)]),
                         label=f'discrete_{i}_{k}')
        for j in range(num_pallets):
            cqm.add_constraint(
                - (2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                   vars.selector[i, k, 0]) * num_pallets * pallets.length +
                (vars.x[i] + sx[i] - vars.x[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_0')

            cqm.add_constraint(
                -(2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                  vars.selector[i, k, 1]) * pallets.width +
                (vars.y[i] + sy[i] - vars.y[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_1')

            cqm.add_constraint(
                -(2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                  vars.selector[i, k, 2]) * pallets.height +
                (vars.z[i] + sz[i] - vars.z[k]) <= 0,
                label=f'overlap_{i}_{k}_{j}_2')

            cqm.add_constraint(
                -(2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                  vars.selector[i, k, 3]) * num_pallets * pallets.length +
                (vars.x[k] + sx[k] - vars.x[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_3')

            cqm.add_constraint(
                -(2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                  vars.selector[i, k, 4]) * pallets.width +
                (vars.y[k] + sy[k] - vars.y[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_4')

            cqm.add_constraint(
                -(2 - (vars.pallet_loc[i, j] * vars.pallet_loc[k, j]) -
                  vars.selector[i, k, 5]) * pallets.height +
                (vars.z[k] + sz[k] - vars.z[i]) <= 0,
                label=f'overlap_{i}_{k}_{j}_5')

    for i in range(num_cases):
        cqm.add_constraint(
            quicksum([vars.pallet_loc[i, j] for j in range(num_pallets)]) == 1,
            label=f'box_{i}_max_packed')


def _add_boundary_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                              pallets: Pallets, cases: Cases, origins: list):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins
    for i in range(num_cases):
        for j in range(num_pallets):
            cqm.add_constraint(vars.z[i] + sz[i] - vars.bin_height[j] -
                               (1 - vars.pallet_on[j]) * pallets.height <= 0,
                               label=f'maxx_height_{i}_{j}')

            cqm.add_constraint(vars.x[i] + sx[i] - pallets.length * (j + 1)
                               - (1 - vars.pallet_loc[i, j]) *
                               num_pallets * pallets.length <= 0,
                               label=f'maxx_{i}_{j}_less')

            cqm.add_constraint(
                vars.x[i] - pallets.length * j * vars.pallet_loc[i, j] >= 0,
                label=f'maxx_{i}_{j}_greater')

            cqm.add_constraint(
                (vars.y[i] + sy[i] - pallets.width) -
                (1 - vars.pallet_loc[i, j]) * pallets.width <= 0,
                label=f'maxy_{i}_{j}_less')

            cqm.add_constraint(
                (vars.z[i] + sz[i] - pallets.height) -
                (1 - vars.pallet_loc[i, j]) * pallets.height <= 0,
                label=f'maxz_{i}_{j}_less')


def _define_objective(cqm: ConstrainedQuadraticModel, vars: Variables,
                      pallets: Pallets, cases: Cases, origins: list):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins

    # First term of objective: minimize average height of boxes
    first_obj_term = quicksum(
        vars.z[i] + sz[i] for i in range(num_cases)) / num_cases

    # Second term of objective: minimize height of the box at the top of the bin
    second_obj_term = quicksum(vars.bin_height[j] for j in range(num_pallets))

    # Third term of the objective:
    pallet_available_space = [
        pallets.length * pallets.width * pallets.height * vars.pallet_on[j]
        for j in range(num_pallets)]
    boxes_used_space = [cases.length[i] * cases.width[i] * cases.height[i] *
                        vars.pallet_loc[i, j] for i in range(num_cases)
                        for j in range(num_pallets)]
    denominator = pallets.height * (pallets.length * pallets.width) ** 2
    third_obj_term = quicksum(
        (pallet_available_space[j] - boxes_used_space[j]) ** 2 / denominator
        for j in range(num_pallets))

    cqm.set_objective(first_obj_term + second_obj_term + third_obj_term)


def build_cqm(vars: Variables, pallets: Pallets,
              cases: Cases) -> Tuple[ConstrainedQuadraticModel, list]:
    """Builds the CQM model from the problem variables and data.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
        cases: Instance of ``Cases``, representing items packed into containers.

    Returns:
        A ``dimod.CQM`` object that defines the 3D bin packing problem.
        origins: List of case dimensions based on orientations of cases.
    
    """
    cqm = ConstrainedQuadraticModel()
    origins = _add_orientation_constraints(cqm, vars, cases)
    _add_pallet_on_constraint(cqm, vars, pallets, cases)
    _add_geometric_constraints(cqm, vars, pallets, cases, origins)
    _add_boundary_constraints(cqm, vars, pallets, cases, origins)
    _define_objective(cqm, vars, pallets, cases, origins)

    return cqm, origins


def call_cqm_solver(cqm: ConstrainedQuadraticModel,
                    time_limit: float) -> SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``CQM`` object that defines the 3D bin packing problem.
        time_limit: Time limit parameter to pass on to the CQM sampler.

    Returns:
        A ``dimod.SampleSet`` that represents the best feasible solution found.
    
    """
    sampler = HSSCQMSampler()
    t0 = time.perf_counter()
    res = sampler.sample(cqm, time_limit=time_limit)
    res.resolve()
    feasible_sampleset = res.filter(lambda d: d.is_feasible)
    print(feasible_sampleset)
    best_feasible = feasible_sampleset.first.sample
    t = time.perf_counter() - t0
    print(f'Time: {t} s')

    return best_feasible


def print_results(cqm: ConstrainedQuadraticModel, vars: Variables,
                  sample: SampleSet,
                  cases: Cases, pallets: Pallets, origins: list):
    """Helper function to print results of CQM sampler.

    Args:
        cqm: A ``dimod.CQM`` object that defines the 3D bin packing problem.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        cases: Instance of ``Cases``, representing items packed into containers.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
        origins: List of case dimensions based on orientations of cases.
    
    """
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins
    print(f'Objective: {cqm.objective.energy(sample)}')
    vs = {i: (
        sum((j + 1) * vars.pallet_loc[i, j].energy(sample) for j in
            range(num_pallets)),
        np.round(vars.x[i].energy(sample), 2),
        np.round(vars.y[i].energy(sample), 2),
        np.round(vars.z[i].energy(sample), 2),
        np.round(sx[i].energy(sample), 2),
        np.round(sy[i].energy(sample), 2),
        np.round(sz[i].energy(sample), 2)) for i in range(num_cases)}
    for k, v in vs.items():
        print(k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_filepath", type=str, nargs="?",
                        help="Filepath to bin-packing data file.",
                        default="input/sample_data.txt")
    parser.add_argument("pallet_length", type=int, nargs="?",
                        help="Length dimension of pallet.",
                        default=100)
    parser.add_argument("pallet_width", type=int, nargs="?",
                        help="Width dimension of pallet.",
                        default=100)
    parser.add_argument("pallet_height", type=int, nargs="?",
                        help="Height dimension of pallet.",
                        default=110)
    parser.add_argument("num_pallets", type=int, nargs="?",
                        help="Specify number of pallets to pack.",
                        default=1)
    parser.add_argument("time_limit", type=float, nargs="?",
                        help="Time limit for the hybrid CQM Solver to run in"
                             " seconds.",
                        default=20)
    args = parser.parse_args()
    time_limit = args.time_limit
    cases = Cases(args.data_filepath)
    pallets = Pallets(length=args.pallet_length,
                      width=args.pallet_width,
                      height=args.pallet_height,
                      num_pallets=args.num_pallets,
                      cases=cases)

    vars = Variables(cases, pallets)

    cqm, origins = build_cqm(vars, pallets, cases)

    print_cqm_stats(cqm)

    best_feasible = call_cqm_solver(cqm, time_limit)

    if len(best_feasible) > 0:
        print_results(cqm, vars, best_feasible, cases, pallets,
                      origins)
        plot_cuboids(best_feasible, vars, cases, pallets, origins)
    else:
        print("No feasible solution found this run.")
