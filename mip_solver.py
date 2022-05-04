import itertools
import time
import typing

import mip
import dimod


class MIPCQMSolver:
    @staticmethod
    def mip_vartype(vartype: dimod.typing.VartypeLike) -> str:
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
    def qm_to_expression(qm: typing.Union[dimod.QuadraticModel, dimod.BinaryQuadraticModel],
                         variable_map: typing.Dict[dimod.typing.Variable, mip.Var],
                         ) -> mip.LinExpr:
        if not qm.is_linear:
            raise ValueError("MIP cannot support quadratic interactions")
        return mip.xsum(itertools.chain(
            (variable_map[v] * bias for v, bias in qm.iter_linear()),
            (qm.offset,)
            ))

    @classmethod
    def sample_cqm(cls, cqm: dimod.ConstrainedQuadraticModel,
                   time_limit: float = float('inf'),
                   ) -> dimod.SampleSet:

        model = mip.Model()

        variable_map: typing.Dict[dimod.typing.Variable, mip.Var] = dict()
        for v in cqm.variables:
            var = model.add_var(
                name=v,
                lb=cqm.lower_bound(v),
                ub=cqm.upper_bound(v),
                var_type=cls.mip_vartype(cqm.vartype(v))
                )
            variable_map[v] = var

        model.objective = cls.qm_to_expression(cqm.objective, variable_map)

        for label, constraint in cqm.constraints.items():
            lhs = cls.qm_to_expression(constraint.lhs, variable_map)
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
