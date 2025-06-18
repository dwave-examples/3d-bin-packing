# Copyright 2024 D-Wave Systems Inc.
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

import time
import typing

import dimod
import numpy as np
import scipy.optimize


class SciPyCQMSolver:
    """An Ocean wrapper for SciPy's MILP HiGHS solver.

    See :func:`scipy.optimize.milp()`
    """

    @staticmethod
    def iter_constraints(
        cqm: dimod.ConstrainedQuadraticModel,
    ) -> typing.Iterator[scipy.optimize.LinearConstraint]:
        num_variables = cqm.num_variables()

        for comp in cqm.constraints.values():
            if comp.sense is dimod.sym.Sense.Eq:
                lb = ub = comp.rhs - comp.lhs.offset  # move offset (if not 0) to rhs of constraint
            elif comp.sense is dimod.sym.Sense.Ge:
                lb = comp.rhs - comp.lhs.offset
                ub = +float("inf")
            elif comp.sense is dimod.sym.Sense.Le:
                lb = -float("inf")
                ub = comp.rhs - comp.lhs.offset
            else:
                raise ValueError("unexpected constraint sense")

            A = np.zeros(num_variables, dtype=float)
            for v, bias in comp.lhs.linear.items():
                A[cqm.variables.index(v)] = bias  # variables.index() is O(1)

            # Create the LinearConstraint.
            # We save A as a csr matrix to save on a bit of memory
            yield scipy.optimize.LinearConstraint(scipy.sparse.csr_array([A]), lb=lb, ub=ub)

    @staticmethod
    def sample_cqm(
        cqm: dimod.ConstrainedQuadraticModel,
        time_limit: float = float("inf"),
    ) -> dimod.SampleSet:
        """Use HiGHS via SciPy to solve a constrained quadratic model.

        Note that HiGHS requires the objective and constraints to be
        linear.

        Args:
            cqm: A constrained quadratic model.
            time_limit: The maximum time in seconds to search.

        Returns:
            A sample set with any solutions returned by
            :func:`scipy.optimize.milp()`.

        Raises:
            ValueError: If the given constrained quadratic model contains
                any quadratic terms.

        """
        # Note: we name the input variables according to SciPy's naming
        # conventions

        # Handle the empty case
        if not cqm.variables:
            return dimod.SampleSet.from_samples_cqm([], cqm, info=dict(run_time=0))

        # Check that we're a linear model
        if not cqm.objective.is_linear():
            raise ValueError(
                "scipy.optimize.milp() does not support objectives " "with quadratic interactions"
            )
        if not all(comp.lhs.is_linear() for comp in cqm.constraints.values()):
            raise ValueError(
                "scipy.optimize.milp() does not support constraints " "with quadratic interactions"
            )

        num_variables = cqm.num_variables()

        # The objective
        c = np.empty(num_variables, dtype=float)
        for i, v in enumerate(cqm.variables):
            c[i] = cqm.objective.linear.get(v, 0)

        # The vartypes and the bounds
        integrality = np.empty(num_variables, dtype=np.uint8)
        lb = np.empty(num_variables, dtype=float)
        ub = np.empty(num_variables, dtype=float)
        for i, v in enumerate(cqm.variables):
            vartype = cqm.vartype(v)
            if vartype is dimod.BINARY:
                integrality[i] = 1  # 1 indicates integer variable
            elif vartype is dimod.INTEGER:
                integrality[i] = 1  # 1 indicates integer variable
            elif vartype is dimod.REAL:
                integrality[i] = 0  # 0 indicates continuous
            else:
                raise ValueError("unexpected vartype")

            lb[i] = cqm.lower_bound(v)
            ub[i] = cqm.upper_bound(v)

        # The constraints
        constraints = list(SciPyCQMSolver.iter_constraints(cqm))

        t = time.perf_counter()
        solution = scipy.optimize.milp(
            c,
            integrality=integrality,
            bounds=scipy.optimize.Bounds(lb=lb, ub=ub),
            options=dict(time_limit=time_limit),
            constraints=constraints,
        )
        run_time = time.perf_counter() - t

        # If we're infeasible, return an empty solution
        if solution.x is None:
            return dimod.SampleSet.from_samples_cqm([], cqm, info=dict(run_time=run_time))

        # Otherwise we can just read the solution out and convert it into a
        # dimod sampleset
        sampleset = dimod.SampleSet.from_samples_cqm(
            (solution.x, cqm.variables), cqm, info=dict(run_time=run_time)
        )

        return sampleset
