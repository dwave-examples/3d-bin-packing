import argparse
from email.policy import default
from itertools import combinations
import numpy as np
import os
import time
import dimod

from utils import print_cqm_stats, plot_cuboids

# todo: remove this before launch
use_local = True
if use_local:
    from cqmsolver.hss_sampler import HSSCQMSampler
else:
    from dwave.system import LeapHybridSampler


class Cases:
    """Class for representing cuboid item data in a 3d-bin packing problem.

    Args:
        data_filepath: Specifies file to load comma-delimited data from.
            See sample_data.txt for format.
    
    """
    def __init__(self, data_filepath: str):
        path = os.path.join(os.path.dirname(__file__), data_filepath)
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int32)
        self.case_ids = data["case_id"]
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)      
        self.lengths = np.repeat(data["length"], data["quantity"])
        self.widths = np.repeat(data["width"], data["quantity"])
        self.heights = np.repeat(data["height"], data["quantity"])
        print(f'Number of cases: {self.num_cases}')


class Pallets:
    """Class for representing cuboid container data in a 3d-bin packing problem.

    Args:
        length: Length of container(s).
        width: Width of container(s).
        height: Height of container(s).
        num_pallets: Number of containers.
        cases: Instance of ``Cases``, representing items packed into containers.

    """
    def __init__(self, length: int, width: int, height: int, num_pallets: int, cases: Cases):
        self.length = length
        self.width = width
        self.height = height
        self.num_pallets = num_pallets
        self.lowest_num_pallet = np.ceil(
            np.sum(cases.lengths * cases.widths * cases.heights) / (length * width * height))
        assert self.lowest_num_pallet <= num_pallets, \
            f'number of pallets is at least {self.lowest_num_pallet}'
        print(f'Minimum Number of pallets required: {self.lowest_num_pallet}')


class Variables:
    """Class that collects all CQM model variables for the 3d-bin packing problem.

    Args:
        cases: Instance of ``Cases``, representing items packed into containers.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
    
    """
    def __init__(self, cases: Cases, pallets: Pallets):
        num_cases = cases.num_cases
        num_pallets = pallets.num_pallets
        self.x = {i: dimod.Real(f'x_{i}',
                                lower_bound=0,
                                upper_bound=pallets.length * pallets.num_pallets)
                  for i in range(num_cases)}
        self.y = {i: dimod.Real(f'y_{i}', lower_bound=0, upper_bound=pallets.width)
                  for i in range(num_cases)}
        self.z = {i: dimod.Real(f'z_{i}', lower_bound=0, upper_bound=pallets.height)
                  for i in range(num_cases)}

        self.zH = {
            j: dimod.Real(label=f'upper_bound_{j}', upper_bound=pallets.height)
            for j in range(num_pallets)}

        self.pallet_loc = {(i, j): dimod.Binary(f'x{i}_IN_{j}')
                           for i in range(num_cases)
                           for j in range(num_pallets)}

        self.pallet_on = {i: dimod.Binary(f'x{i}_is_in')
                          for i in range(num_pallets)}

        self.o = {i: dimod.Binary(f'o_{i}') for i in range(num_cases)}

        self.selector = {(i, j, k): dimod.Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}


def _add_pallet_on_constraint(cqm: dimod.CQM, vars: Variables, pallets: Pallets, cases: Cases):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    for j in range(num_pallets):
        cqm.add_constraint((1 - vars.pallet_on[j]) * dimod.quicksum(
            vars.pallet_loc[i, j] for i in range(num_cases)) <= 0, label=f'p_on_{j}')


def define_case_dimensions(vars: Variables, cases: Cases) -> list:
    """Define case dimensions based on orientations in the length or width dimensions.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        cases: Instance of ``Cases``, representing items packed into containers.
    
    Returns:
        List of case dimensions based on orientations of cases.

    """
    num_cases = cases.num_cases
    ox = {}
    oy = {}
    oz = {}
    for i in range(num_cases):
        ox[i] = cases.lengths[i] * vars.o[i] + cases.widths[i] * (1 - vars.o[i])
        oy[i] = cases.widths[i] * vars.o[i] + cases.lengths[i] * (1 - vars.o[i])
        oz[i] = cases.heights

    return [ox, oy, oz]


def _add_positional_constraints(cqm: dimod.CQM, vars: Variables, pallets: Pallets, cases: Cases, origins: list):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins
    for i, j in combinations(range(num_cases), r=2):
        cqm.add_discrete([f'sel_{i}_{j}_{k}' for k in range(6)],
                         label=f'discrete_{i}_{j}')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 0]) * num_pallets * pallets.length +
            (vars.x[i] + sx[i] - vars.x[j]) <= 0,
            label=f'overlap_{i}_{j}_0')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 1]) * pallets.width +
            (vars.y[i] + sy[i] - vars.y[j]) <= 0,
            label=f'overlap_{i}_{j}_1')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 2]) * pallets.height +
            (vars.z[i] + cases.heights[i] - vars.z[j]) <= 0,
            label=f'overlap_{i}_{j}_2')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 3]) * num_pallets * pallets.length +
            (vars.x[j] + sx[j] - vars.x[i]) <= 0,
            label=f'overlap_{i}_{j}_3')
        cqm.add_constraint(
            -(1 - vars.selector[i, j, 4]) * pallets.width +
            (vars.y[j] + sy[j] - vars.y[i]) <= 0,
            label=f'overlap_{i}_{j}_4')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 5]) * pallets.height +
            (vars.z[j] + cases.heights[j] - vars.z[i]) <= 0,
            label=f'overlap_{i}_{j}_5')

    if num_pallets > 1:
        for i in range(num_cases):
            cqm.add_discrete([f'x{i}_IN_{j}' for j in range(num_pallets)],
                             label=f'c{i}_in_which')


def _add_boundary_constraints(cqm: dimod.CQM, vars: Variables, pallets: Pallets, cases: Cases, origins: list):
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    sx, sy, sz = origins
    for i in range(num_cases):
        for j in range(num_pallets):
            cqm.add_constraint(vars.z[i] + cases.heights[i] - vars.zH[j] <= 0,
                               label=f'maxx_height_{i}_{j}')

            cqm.add_constraint(vars.x[i] + sx[i] - pallets.length * (j + 1)
                               - (1 - vars.pallet_loc[i, j]) * num_pallets * pallets.length <= 0,
                               label=f'maxx_{i}_{j}_less')
            cqm.add_constraint(vars.x[i] - pallets.length * j * vars.pallet_loc[i, j] >= 0,
                               label=f'maxx_{i}_{j}_greater')
            cqm.add_constraint(
                (vars.y[i] + sy[i] - pallets.width) -
                (1 - vars.pallet_loc[i, j]) * pallets.width <= 0,
                label=f'maxy_{i}_{j}_less')
            cqm.add_constraint(
                (vars.z[i] + cases.heights[i] - pallets.height) -
                (1 - vars.pallet_loc[i, j]) * pallets.height <= 0,
                label=f'maxz_{i}_{j}_less')


def _define_objective(cqm: dimod.CQM, vars: Variables, pallets: Pallets, cases: Cases):
    pallet_count = dimod.quicksum(vars.pallet_on.values())
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    case_height = dimod.quicksum(vars.z[i] +
                                 cases.heights[i] for i in range(num_cases)) / num_cases
    cqm.set_objective(dimod.quicksum(vars.zH[j] for j in range(num_pallets))
                      + case_height + 100 * pallet_count)


def build_cqm(vars: Variables, pallets: Pallets, cases: Cases, origins: list) -> dimod.CQM:
    """Builds the CQM model from the problem variables and data.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
        cases: Instance of ``Cases``, representing items packed into containers.
        origins: List of case dimensions based on orientations of cases.

    Returns:
        A ``dimod.CQM`` object that defines the 3D bin packing problem.
    
    """
    cqm = dimod.ConstrainedQuadraticModel()
    _add_pallet_on_constraint(cqm, vars, pallets, cases)
    _add_positional_constraints(cqm, vars, pallets, cases, origins)
    _add_boundary_constraints(cqm, vars, pallets, cases, origins)
    _define_objective(cqm, vars, pallets, cases)
    
    return cqm


def call_cqm_solver(cqm: dimod.CQM, time_limit: float) -> dimod.SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``dimod.CQM`` object that defines the 3D bin packing problem.
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


def print_results(cqm: dimod.CQM, vars: Variables, sample: dimod.SampleSet, 
                  cases: Cases, pallets: Pallets, origins: list):
    """Helper function to print results of CQM sampler.

    Args:
        cqm: A ``dimod.CQM`` object that defines the 3D bin packing problem.
        vars:Instance of ``Variables`` that defines the complete set of variables
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
    vs = {i: (np.round(vars.x[i].energy(sample), 2),
            np.round(vars.y[i].energy(sample), 2),
            np.round(vars.z[i].energy(sample), 2),
            np.round(sx[i].energy(sample), 2),
            np.round(sy[i].energy(sample), 2),
            cases.heights[i])
            for i in range(num_cases) for j in range(num_pallets)}
    solution_height = vars.zH[0].energy(sample)
    for k, v in vs.items():
        print(k, v)
    print("height: ", solution_height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_filepath", type=str, nargs="?",
                        help="Filepath to bin-packing data file.",
                        default="sample_data.txt")
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
                        help="Time limit for the hybrid CQM Solver to run in seconds.",
                        default=20)
    args = parser.parse_args()

    cases = Cases(args.data_filepath)
    pallets = Pallets(length=args.pallet_length,
                      width=args.pallet_width,
                      height=args.pallet_height,
                      num_pallets=args.num_pallets,
                      cases=cases)
    time_limit = 20
    vars = Variables(cases, pallets)
    origins = define_case_dimensions(vars, cases)

    cqm = build_cqm(vars, pallets, cases, origins)

    print_cqm_stats(cqm)

    best_feasible = call_cqm_solver(cqm, time_limit)

    if len(best_feasible) > 0:
        print_results(cqm, vars, best_feasible, cases, pallets,
                      origins)
        plot_cuboids(best_feasible, vars, cases, pallets, origins)
