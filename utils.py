import plotly.graph_objects as go
import numpy as np
from tabulate import tabulate
from typing import List, Optional, TYPE_CHECKING
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
    print(tabulate([["Binary", "Integer", "Continuous", "Quad", "Linear",
                     "One-hot", "EQ  ", "LT", "GT"],
                    [num_binaries, num_integers, num_continuous,
                     num_quadratic_constraints,
                     num_linear_constraints, num_discretes,
                     num_equality_constraints,
                     num_le_inequality_constraints,
                     num_ge_inequality_constraints]],
                   headers="firstrow"))


def _cuboid_data2(o: tuple, size: tuple = (1, 1, 1)):
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


def _plotCubeAt2(positions: List[tuple], sizes: Optional[List[tuple]] = None,
                 colors: Optional[List[str]] = None, **kwargs) -> list:
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(
        positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(
        positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        box_points = _cuboid_data2(p, size=s)
        # Get all unique vertices for 3d Mesh
        x, y, z = np.unique(np.vstack(box_points), axis=0).T
        g.append(go.Mesh3d(x=x, y=y, z=z, **kwargs))
    
    return g


def _plot_cuboid(positions: List[tuple], sizes: List[tuple],
                 pallet_length: int,
                 pallet_width: int, pallet_height: int, **kwargs) -> go.Figure:
    colors = [[tuple(list(np.random.rand(3)) + [0.1])] * 6 for i in
              range(len(positions))]
    colors = np.vstack(colors)

    num_pallets = _plotCubeAt2(positions, sizes, **kwargs)
    fig = go.Figure(data=num_pallets)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0,pallet_length*1.1]),
        yaxis=dict(range=[0,pallet_width*1.1]),
        zaxis=dict(range=[0,pallet_height*1.1])
    ))

    return fig


def plot_cuboids(sample: dimod.SampleSet, vars: "Variables", cases: "Cases",
                 pallets: "Pallets", origins: list, **kwargs) -> go.Figure:
    """Visualization utility tool to view 3D bin packing solution.

    Args:
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        cases: Instance of ``Cases``, representing items packed into containers.
        pallets: Instance of ``Pallets``, representing containers to pack items into.
        origins: List of case dimensions based on orientations of cases.
    
    Returns:
        ``plotly.graph_objects.Figure`` with all items packed according to CQM results.
    
    """
    sx, sy, sz = origins
    num_cases = cases.num_cases
    num_pallets = pallets.num_pallets
    positions = []
    sizes = []
    for i in range(num_cases):
        if sum(vars.pallet_loc[i, j].energy(sample) for j in
               range(num_pallets)):
            positions.append(
                (vars.x[i].energy(sample), vars.y[i].energy(sample),
                 vars.z[i].energy(sample)))
            sizes.append((sx[i].energy(sample),
                          sy[i].energy(sample),
                          cases.height[i]))
    fig = _plot_cuboid(positions, sizes, pallets.length*num_pallets, 
                       pallets.width, pallets.height, **kwargs)

    return fig
