import os
import time
from itertools import combinations
import numpy as np
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
        filepath: Specifies file to load comma-delimited data from.
            See sample_data.txt for format.
    
    """
    def __init__(self, filepath: str):
        path = os.path.join(os.path.dirname(__file__), filepath)
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
        nc = cases.num_cases
        pc = pallets.num_pallets
        self.x = {i: dimod.Real(f'x_{i}',
                                lower_bound=0,
                                upper_bound=pallets.length * pallets.num_pallets)
                  for i in range(nc)}
        self.y = {i: dimod.Real(f'y_{i}', lower_bound=0, upper_bound=pallets.width)
                  for i in range(nc)}
        self.z = {i: dimod.Real(f'z_{i}', lower_bound=0, upper_bound=pallets.height)
                  for i in range(nc)}

        self.zH = {
            j: dimod.Real(label=f'upper_bound_{j}', upper_bound=pallets.height)
            for j in range(pc)}

        self.pallet_loc = {(i, j): dimod.Binary(f'x{i}_IN_{j}')
                           for i in range(nc)
                           for j in range(pc)}

        self.pallet_on = {i: dimod.Binary(f'x{i}_is_in')
                          for i in range(pc)}

        self.o = {i: dimod.Binary(f'o_{i}') for i in range(nc)}

        self.selector = {(i, j, k): dimod.Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(nc), r=2)
                         for k in range(6)}


def add_pallet_on_constraint(cqm, vars, pallets, cases):
    nc = cases.num_cases
    pc = pallets.num_pallets
    for j in range(pc):
        cqm.add_constraint((1 - vars.pallet_on[j]) * dimod.quicksum(
            vars.pallet_loc[i, j] for i in range(nc)) <= 0, label=f'p_on_{j}')


def define_case_dimensions(vars, cases):
    nc = cases.num_cases
    ox = {}
    oy = {}
    oz = {}
    for i in range(nc):
        ox[i] = cases.lengths[i] * vars.o[i] + cases.widths[i] * (1 - vars.o[i])
        oy[i] = cases.widths[i] * vars.o[i] + cases.lengths[i] * (1 - vars.o[i])
        oz[i] = cases.heights

    return [ox, oy, oz]


def add_positional_constraints(cqm, vars, pallets, cases, origins):
    nc = cases.num_cases
    pc = pallets.num_pallets
    sx, sy, sz = origins
    for i, j in combinations(range(nc), r=2):
        cqm.add_discrete([f'sel_{i}_{j}_{k}' for k in range(6)],
                         label=f'discrete_{i}_{j}')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 0]) * pc * pallets.length +
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
            -(1 - vars.selector[i, j, 3]) * pc * pallets.length +
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

    if pc > 1:
        for i in range(nc):
            cqm.add_discrete([f'x{i}_IN_{j}' for j in range(pc)],
                             label=f'c{i}_in_which')


def add_boundary_constraints(cqm, vars, pallets, cases, origins):
    nc = cases.num_cases
    pc = pallets.num_pallets
    sx, sy, sz = origins
    for i in range(nc):
        for j in range(pc):
            cqm.add_constraint(vars.z[i] + cases.heights[i] - vars.zH[j] <= 0,
                               label=f'maxx_height_{i}_{j}')

            cqm.add_constraint(vars.x[i] + sx[i] - pallets.length * (j + 1)
                               - (1 - vars.pallet_loc[i, j]) * pc * pallets.length <= 0,
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


def define_objective(cqm, vars, cases, pallets):
    pallet_count = dimod.quicksum(vars.pallet_on.values())
    nc = cases.num_cases
    pc = pallets.num_pallets
    case_height = dimod.quicksum(vars.z[i] +
                                 cases.heights[i] for i in range(nc)) / nc
    cqm.set_objective(dimod.quicksum(vars.zH[j] for j in range(pc))
                      + case_height + 100 * pallet_count)


def call_cqm_solver(cqm, time_limit):
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


def print_results(cqm, vars, sample, cases, pallets, origins):
        nc = cases.num_cases
        pc = pallets.num_pallets
        sx, sy, sz = origins
        print(f'Objective: {cqm.objective.energy(sample)}')
        vs = {i: (np.round(vars.x[i].energy(sample), 2),
              np.round(vars.y[i].energy(sample), 2),
              np.round(vars.z[i].energy(sample), 2),
              np.round(sx[i].energy(sample), 2),
              np.round(sy[i].energy(sample), 2),
              cases.heights[i])
              for i in range(nc) for j in range(pc)}
        solution_height = vars.zH[0].energy(sample)
        for k, v in vs.items():
            print(k, v)
        print("height: ", solution_height)


if __name__ == '__main__':
    # Get length, width and height of the boxes
    cases = Cases("sample_data.txt")
    pallets = Pallets(length=100, width=100, height=110, num_pallets=1,
                     cases=cases)
    time_limit = 20
    vars = Variables(cases, pallets)
    cqm = dimod.ConstrainedQuadraticModel()
    add_pallet_on_constraint(cqm, vars, pallets, cases)

    origins = define_case_dimensions(vars, cases)

    add_positional_constraints(cqm, vars, pallets, cases, origins)

    add_boundary_constraints(cqm, vars, pallets, cases, origins)

    define_objective(cqm, vars, cases, pallets)

    print_cqm_stats(cqm)

    best_feasible = call_cqm_solver(cqm, time_limit)

    if len(best_feasible) > 0:
        print_results(cqm, vars, best_feasible, cases, pallets,
                      origins)
        plot_cuboids(best_feasible, vars, cases, pallets, origins)
