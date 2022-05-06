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

import unittest

import dimod

from mip_solver import MIPCQMSolver


class TestMIPCQMSolver(unittest.TestCase):
    def test_empty(self):
        cqm = dimod.ConstrainedQuadraticModel()
        sampleset = MIPCQMSolver().sample_cqm(cqm)
        self.assertEqual(len(sampleset), 0)

    def test_infease(self):
        cqm = dimod.ConstrainedQuadraticModel()

        i, j = dimod.Integers('ij')

        cqm.add_constraint(i - j <= -1)
        cqm.add_constraint(i - j >= +1)

        sampleset = MIPCQMSolver().sample_cqm(cqm)
        self.assertEqual(len(sampleset), 0)

    def test_bounds(self):
        cqm = dimod.ConstrainedQuadraticModel()

        i = dimod.Integer('i', lower_bound=-5, upper_bound=5)

        cqm.set_objective(i)
        sampleset = MIPCQMSolver().sample_cqm(cqm)
        self.assertEqual(sampleset.first.sample['i'], -5)

        cqm.set_objective(-i)
        sampleset = MIPCQMSolver().sample_cqm(cqm)
        self.assertEqual(sampleset.first.sample['i'], 5)

    def test_quadratic(self):
        cqm = dimod.ConstrainedQuadraticModel()

        i, j = dimod.Integers('ij')

        cqm.add_constraint(i*j <= 5)

        with self.assertRaises(ValueError):
            MIPCQMSolver().sample_cqm(cqm)

    def test_vartypes(self):

        cqm = dimod.ConstrainedQuadraticModel()

        i = dimod.Integer('i')
        a = dimod.Real('a')
        x = dimod.Binary('x')
        s = dimod.Spin('s')

        cqm.set_objective(-i - a - x)
        cqm.add_constraint(i <= 5.5)
        cqm.add_constraint(a <= 6.5)
        cqm.add_constraint(x <= 7.5)

        sampleset = MIPCQMSolver().sample_cqm(cqm)

        self.assertEqual(sampleset.first.sample, {'i': 5, 'a': 6.5, 'x': 1})

        cqm.add_constraint(s <= 5)
        with self.assertRaises(ValueError):
            MIPCQMSolver().sample_cqm(cqm)
