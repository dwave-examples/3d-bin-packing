import dimod
from itertools import combinations
import numpy as np
import time
from utils import print_cqm_stats, plot_cuboids

# todo: remove this before launch
use_local = True
if use_local:
    from cqmsolver.hss_sampler import HSSCQMSampler
else:
    from dwave.system import LeapHybridSampler


class Cases:
    def __init__(self, case_dim_lower, case_dim_higher, num_items):
        self.l = np.array([np.random.randint(case_dim_lower, case_dim_higher)
                           for _ in range(num_items)])
        self.w = np.array([np.random.randint(case_dim_lower, case_dim_higher)
                           for _ in range(num_items)])
        self.h = np.array([np.random.randint(case_dim_lower, case_dim_higher)
                           for _ in range(num_items)])
        self.num_items = num_items
        print(f'Number of cases: {self.num_items}')


class Pallet:
    def __init__(self, length, width, height, num_pallets, cases):
        self.l = length
        self.w = width
        self.h = height
        self.num_pallets = num_pallets
        self.lowest_num_pallet = np.ceil(
            np.sum(cases.l * cases.w * cases.h) / (length * width * height))
        assert self.lowest_num_pallet <= num_pallets, \
            f'number of pallets is at least {self.lowest_num_pallet}'
        print(f'Minimum Number of pallets required: {self.lowest_num_pallet}')


class Variables:
    def __init__(self, cases, pallets):
        nc = cases.num_items
        pc = pallets.num_pallets
        self.x = {i: dimod.Real(f'x_{i}',
                                lower_bound=0,
                                upper_bound=pallets.l * pallets.num_pallets)
                  for i in range(nc)}
        self.y = {i: dimod.Real(f'y_{i}', lower_bound=0, upper_bound=pallets.w)
                  for i in range(nc)}
        self.z = {i: dimod.Real(f'z_{i}', lower_bound=0, upper_bound=pallets.h)
                  for i in range(nc)}

        self.zH = {
            j: dimod.Real(label=f'upper_bound_{j}', upper_bound=pallets.h)
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
    nc = cases.num_items
    pc = pallets.num_pallets
    for j in range(pc):
        cqm.add_constraint((1 - vars.pallet_on[j]) * dimod.quicksum(
            vars.pallet_loc[i, j] for i in range(nc)) <= 0, label=f'p_on_{j}')


def define_case_dimensions(vars, cases):
    nc = cases.num_items
    ox = {}
    oy = {}
    oz = {}
    for i in range(nc):
        ox[i] = cases.l[i] * vars.o[i] + cases.w[i] * (1 - vars.o[i])
        oy[i] = cases.w[i] * vars.o[i] + cases.l[i] * (1 - vars.o[i])
        oz[i] = cases.h

    return [ox, oy, oz]


def add_positional_constraints(cqm, vars, pallets, cases, origins):
    nc = cases.num_items
    pc = pallets.num_pallets
    sx, sy, sz = origins
    for i, j in combinations(range(nc), r=2):
        cqm.add_discrete([f'sel_{i}_{j}_{k}' for k in range(6)],
                         label=f'discrete_{i}_{j}')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 0]) * pc * pallets.l +
            (vars.x[i] + sx[i] - vars.x[j]) <= 0,
            label=f'overlap_{i}_{j}_0')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 1]) * pallets.w +
            (vars.y[i] + sy[i] - vars.y[j]) <= 0,
            label=f'overlap_{i}_{j}_1')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 2]) * pallets.h +
            (vars.z[i] + cases.h[i] - vars.z[j]) <= 0,
            label=f'overlap_{i}_{j}_2')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 3]) * pc * pallets.l +
            (vars.x[j] + sx[j] - vars.x[i]) <= 0,
            label=f'overlap_{i}_{j}_3')
        cqm.add_constraint(
            -(1 - vars.selector[i, j, 4]) * pallets.w +
            (vars.y[j] + sy[j] - vars.y[i]) <= 0,
            label=f'overlap_{i}_{j}_4')

        cqm.add_constraint(
            -(1 - vars.selector[i, j, 5]) * pallets.h +
            (vars.z[j] + cases.h[j] - vars.z[i]) <= 0,
            label=f'overlap_{i}_{j}_5')

    if pc > 1:
        for i in range(nc):
            cqm.add_discrete([f'x{i}_IN_{j}' for j in range(pc)],
                             label=f'c{i}_in_which')


def add_boundary_constraints(cqm, vars, pallets, cases, origins):
    nc = cases.num_items
    pc = pallets.num_pallets
    sx, sy, sz = origins
    for i in range(nc):
        for j in range(pc):
            cqm.add_constraint(vars.z[i] + cases.h[i] - vars.zH[j] <= 0,
                               label=f'maxx_height_{i}_{j}')

            cqm.add_constraint(vars.x[i] + sx[i] - pallets.l * (j + 1)
                               - (1 - vars.pallet_loc[i, j]) * pc * pallets.l <= 0,
                               label=f'maxx_{i}_{j}_less')
            cqm.add_constraint(vars.x[i] - pallets.l * j * vars.pallet_loc[i, j] >= 0,
                               label=f'maxx_{i}_{j}_greater')
            cqm.add_constraint(
                (vars.y[i] + sy[i] - pallets.w) -
                (1 - vars.pallet_loc[i, j]) * pallets.w <= 0,
                label=f'maxy_{i}_{j}_less')
            cqm.add_constraint(
                (vars.z[i] + cases.h[i] - pallets.h) -
                (1 - vars.pallet_loc[i, j]) * pallets.h <= 0,
                label=f'maxz_{i}_{j}_less')


def define_objective(cqm, vars, cases, pallets):
    pallet_count = dimod.quicksum(vars.pallet_on.values())
    nc = cases.num_items
    pc = pallets.num_pallets
    case_height = dimod.quicksum(vars.z[i] +
                                 cases.h[i] for i in range(nc)) / nc
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
        nc = cases.num_items
        pc = pallets.num_pallets
        sx, sy, sz = origins
        print(f'Objective: {cqm.objective.energy(sample)}')
        vs = {i: (np.round(vars.x[i].energy(sample), 2),
              np.round(vars.y[i].energy(sample), 2),
              np.round(vars.z[i].energy(sample), 2),
              np.round(sx[i].energy(sample), 2),
              np.round(sy[i].energy(sample), 2),
              cases.h[i])
              for i in range(nc) for j in range(pc)}
        solution_height = vars.zH[0].energy(sample)
        for k, v in vs.items():
            print(k, v)
        print("height: ", solution_height)


if __name__ == '__main__':
    np.random.seed(111)
    # Get length, width and height of the boxes
    cases = Cases(case_dim_lower=5, case_dim_higher=50, num_items=20)
    pallets = Pallet(length=100, width=100, height=110, num_pallets=1,
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
