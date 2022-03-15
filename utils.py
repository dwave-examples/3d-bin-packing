import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from tabulate import tabulate
from typing import TYPE_CHECKING
import dimod

if TYPE_CHECKING:
    from packing3d import Cases, Pallets, Variables

def print_cqm_stats(cqm: dimod.ConstrainedQuadraticModel) -> None:
    """Print some information about the CQM model defining the 3D bin packing problem.

    Args:
        cqm: a dimod cqm model (dimod.cqm)

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

    print(" \n" + "=" * 35 + "MODEL INFORMATION" + "=" * 35)
    print(
        ' ' * 10 + 'Variables' + " " * 20 + 'Constraints' + " " * 15 +
        'Sensitivity')
    print('-' * 30 + " " + '-' * 28 + ' ' + '-' * 18)
    print(tabulate([["Binary", "Integer", "Continuous",  "Quad", "Linear", "One-hot", "EQ  ",
                     "LT", "GT"],
                    [num_binaries, num_integers, num_continuous,
                     num_quadratic_constraints,
                     num_linear_constraints, num_discretes,
                     num_equality_constraints,
                     num_le_inequality_constraints,
                     num_ge_inequality_constraints]],
                   headers="firstrow"))


def _cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def _plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(
        positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(
        positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(_cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=colors, **kwargs)


def _plot_cuboid(positions, sizes, L, W, H):
    colors = [[tuple(list(np.random.rand(3)) + [0.1])] * 6 for i in range(len(positions))]
    colors = np.vstack(colors)

    ax = plt.axes(projection='3d')
    num_pallets = _plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
    ax.add_collection3d(num_pallets)

    ax.set_xlim([0, L * 1.1])
    ax.set_ylim([0, W * 1.1])
    ax.set_zlim([0, H * 1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((L, W, H))
    return ax


def plot_cuboids(sample: dimod.SampleSet, vars: "Variables", cases: "Cases", 
                 pallets: "Pallets", origins: list):
    ox, oy, oz = origins
    num_cases = cases.num_cases
    np = pallets.num_pallets
    positions = []
    sizes = []
    for i in range(num_cases):
        positions.append((vars.x[i].energy(sample), vars.y[i].energy(sample),
                          vars.z[i].energy(sample)))
        sizes.append((ox[i].energy(sample),
                     oy[i].energy(sample),
                    cases.heights[i]))
    ax = _plot_cuboid(positions, sizes, pallets.length * np, pallets.width, pallets.height)
    for i in range(np):
        ax.plot([pallets.length * (i + 1)] * 2, [0, pallets.width], [0, 0], linewidth=4,
                color='r')
    for angle in range(0, 360, 30):
        ax.view_init(30, angle)
        plt.savefig(f'res_{angle}.png')
        plt.pause(.001)
