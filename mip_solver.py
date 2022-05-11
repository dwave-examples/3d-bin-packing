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

import itertools
import time
import typing

import mip
import dimod


class MIPCQMSolver:
    """An Ocean wrapper for Python-MIP's solver.

    See https://www.python-mip.com/
    """
    @staticmethod
    def _mip_vartype(vartype: dimod.typing.VartypeLike) -> str:
        vartype = dimod.as_vartype(vartype, extended=True)
        if vartype is dimod.SPIN:
            raise ValueError("MIP cannot handle SPIN variables")
        elif vartype is dimod.BINARY:
            return 'B'
        elif vartype is dimod.INTEGER:
            return 'I'
        elif vartype is dimod.REAL:
            return 'C'
        else:
            raise ValueError("unexpected vartype")

    @staticmethod
    def _qm_to_expression(qm: typing.Union[dimod.QuadraticModel, dimod.BinaryQuadraticModel],
                          variable_map: typing.Dict[dimod.typing.Variable, mip.Var],
                          ) -> mip.LinExpr:
        if not qm.is_linear():
            raise ValueError("MIP cannot support quadratic interactions")
        return mip.xsum(itertools.chain(
            (variable_map[v] * bias for v, bias in qm.iter_linear()),
            (qm.offset,)
            ))

    @classmethod
    def sample_cqm(cls, cqm: dimod.ConstrainedQuadraticModel,
                   time_limit: float = float('inf'),
                   ) -> dimod.SampleSet:
        """Use Python-MIP to solve a constrained quadratic model.

        Note that Python-MIP requires the objective and constraints to be
        linear.

        Args:
            cqm: A constrained quadratic model.
            time_limit: The maximum time in seconds to search.

        Returns:
            A sample set with any solutions returned by Python-MIP.

        Raises:
            ValueError: If the given constrained quadratic model contains
                any quadratic terms.

        """
        model = mip.Model()

        variable_map: typing.Dict[dimod.typing.Variable, mip.Var] = dict()
        for v in cqm.variables:
            variable_map[v] = model.add_var(
                name=v,
                lb=cqm.lower_bound(v),
                ub=cqm.upper_bound(v),
                var_type=cls._mip_vartype(cqm.vartype(v))
                )

        model.objective = cls._qm_to_expression(cqm.objective, variable_map)

        for label, constraint in cqm.constraints.items():
            lhs = cls._qm_to_expression(constraint.lhs, variable_map)
            rhs = constraint.rhs
            if constraint.sense is dimod.sym.Sense.Le:
                model.add_constr(lhs <= rhs, name=label)
            elif constraint.sense is dimod.sym.Sense.Ge:
                model.add_constr(lhs >= rhs, name=label)
            elif constraint.sense is dimod.sym.Sense.Eq:
                model.add_constr(lhs == rhs, name=label)
            else:
                raise RuntimeError(f"unexpected sense: {lhs.sense!r}")

        t = time.perf_counter()
        model.optimize(max_seconds=time_limit)
        run_time = time.perf_counter() - t

        samples = [
            [variable_map[v].xi(k) for v in cqm.variables]
            for k in range(model.num_solutions)
            ]

        return dimod.SampleSet.from_samples_cqm(
            (samples, cqm.variables), cqm, info=dict(run_time=run_time))
