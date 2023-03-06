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

from itertools import combinations
import numpy as np
import unittest
from unittest.mock import patch

import dimod
from mip_solver import MIPCQMSolver
import packing3d
from packing3d import build_cqm, Cases, Bins, Variables, call_solver
from packing3d import _add_orientation_constraints, _add_bin_on_constraint
from packing3d import _add_boundary_constraints, _add_geometric_constraints

from utils import read_instance


class TestPacking3d(unittest.TestCase):
    def setUp(self):
        self.data = read_instance("./tests/test_data_1.txt")
        self.cases = Cases(self.data)
        self.bins = Bins(self.data, self.cases)
        self.variables = Variables(self.cases, self.bins)

    def test_cases(self):
        np.testing.assert_array_equal(self.cases.case_ids, np.array([0,1]))
        self.assertEqual(self.cases.num_cases, 2)
        np.testing.assert_array_equal(self.cases.length, np.array([2,3]))
        np.testing.assert_array_equal(self.cases.width, np.array([2,3]))
        np.testing.assert_array_equal(self.cases.height, np.array([2,3]))

    def test_bins(self):
        data = self.data.copy()
        self.assertEqual(self.bins.length, 30)
        self.assertEqual(self.bins.width, 40)
        self.assertEqual(self.bins.height, 50)
        self.assertEqual(self.bins.num_bins, 1)
        self.assertEqual(self.bins.lowest_num_bin, 1)

        # Alter bin dimensions to trigger exception
        data['bin_dimensions'] = [1,1,1]
        with self.assertRaises(RuntimeError):
            Bins(data, self.cases)

    def test_variables(self):
        self.assertEqual(len(self.variables.x), self.cases.num_cases)
        self.assertEqual(len(self.variables.y), self.cases.num_cases)
        self.assertEqual(len(self.variables.z), self.cases.num_cases)
        self.assertEqual(len(self.variables.bin_height), self.bins.num_bins)
        self.assertEqual(
            len(self.variables.bin_loc), self.cases.num_cases * self.bins.num_bins
        )
        self.assertEqual(len(self.variables.bin_on), self.bins.num_bins)
        self.assertEqual(len(self.variables.o), self.cases.num_cases * 6)
        self.assertEqual(
            len(self.variables.selector), 
            len(list(combinations(range(self.cases.num_cases), r=2))) * 6
        )

    def test_add_orientation_constraints(self):
        cqm = dimod.ConstrainedQuadraticModel()
        _add_orientation_constraints(cqm, self.variables, self.cases)
        self.assertEqual(len(cqm.constraints), 2)
        for c, cval in cqm.constraints.items():
            self.assertEqual(cval.lhs.num_variables, 6)
            self.assertEqual(sum((cval.lhs.linear.values())), 6)
            self.assertEqual(cval.sense, dimod.sym.Sense.Eq)

    def test_add_bin_on_constraint(self):
        bins = Bins(self.data, self.cases)
        bins.num_bins = 2
        vars = Variables(self.cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        _add_bin_on_constraint(cqm, vars, bins, self.cases)

        # check a feasible solution
        sample = {
            'bin_0_is_used': 1, 'case_0_in_bin_0': 0, 'case_1_in_bin_0': 1,
            'bin_1_is_used': 1, 'case_0_in_bin_1': 1, 'case_1_in_bin_1': 0}
        self.assertEqual(cqm.check_feasible(sample), True)

        # check sequence bin use constraint
        sample = {
            'bin_0_is_used': 0, 'case_0_in_bin_0': 0, 'case_1_in_bin_0': 0,
            'bin_1_is_used': 1, 'case_0_in_bin_1': 1, 'case_1_in_bin_1': 1}

        self.assertEqual(cqm.check_feasible(sample), False)

        # check that it is not possible to put item in a bin that is off
        sample = {
            'bin_0_is_used': 1, 'case_0_in_bin_0': 1, 'case_1_in_bin_0': 0,
            'bin_1_is_used': 0, 'case_0_in_bin_1': 0, 'case_1_in_bin_1': 1}
        self.assertEqual(cqm.check_feasible(sample), False)

    def test_add_geometric(self):
        cqm = dimod.ConstrainedQuadraticModel()
        effective_dimensions = _add_orientation_constraints(cqm, 
                                                            self.variables,
                                                            self.cases)
        _add_geometric_constraints(cqm, 
                                   self.variables, 
                                   self.bins, 
                                   self.cases, 
                                   effective_dimensions)
        self.assertEqual(len(cqm.constraints), 9)
        original_sample = {k: 0 for k in cqm.variables.copy()}
        original_sample['o_0_0'] = 1
        original_sample['o_1_0'] = 1

        for i, v in enumerate(['x', 'y', 'z']):
            # check feasible configurations
            for k in range(2):
                sample = original_sample.copy()
                sample[f'{v}_{1 - k}'] = 10
                sample[f'sel_0_1_{i + 3 * k}'] = 1
                self.assertTrue(cqm.check_feasible(sample))

                sample = original_sample.copy()
                sample[f'{v}_{k}'] = 10
                sample[f'sel_0_1_{i + 3 * (1 - k)}'] = 1
                self.assertTrue(cqm.check_feasible(sample))

            # check infeasible configurations
            for k in range(2):
                sample = original_sample.copy()
                sample[f'{v}_{1 - k}'] = 10
                sample[f'{v}_{k}'] = 9
                sample[f'sel_0_1_{i + 3 * k}'] = 1
                self.assertFalse(cqm.check_feasible(sample))

            # check selectors
            for j in range(6):
                sample = original_sample.copy()
                sample[f'{v}_1'] = 10
                # check when no selector is enforced solution is infeasible
                self.assertFalse(cqm.check_feasible(sample))

                sample[f'sel_0_1_{i}'] = 1
                sample[f'sel_0_1_{j}'] = 1
                if i == j:
                    # check that at least one selector is enforced
                    self.assertTrue(cqm.check_feasible(sample))

                else:
                    # check that at most one selector is enforced
                    self.assertFalse(cqm.check_feasible(sample))

    def test_add_boundary_constraints(self):
        bins = Bins(self.data, self.cases)
        bins.num_bins = 2
        vars = Variables(self.cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        effective_dimensions = _add_orientation_constraints(cqm, 
                                                            vars, 
                                                            self.cases)
        _add_boundary_constraints(cqm, vars, bins, 
                                  self.cases, effective_dimensions)
        self.assertEqual(len(cqm.constraints), 18)

        original_sample = {k: 0 for k in cqm.variables.copy()}
        original_sample['o_0_0'] = 1
        original_sample['o_1_0'] = 1
        for v in ['x', 'y', 'z']:
            # check boundary when items are assigned to bin 1
            sample = original_sample.copy()
            sample['case_0_in_bin_0'] = 1
            sample['case_1_in_bin_0'] = 1
            sample['upper_bound_0'] = 12
            sample[f'{v}_0'] = 10
            self.assertTrue(cqm.check_feasible(sample))

            # # check boundary when items are assigned to bins 1 and 2
            sample = original_sample.copy()
            sample[f'{v}_0'] = 35
            sample[f'x_0'] = 35
            sample['case_0_in_bin_1'] = 1
            sample['case_1_in_bin_0'] = 1
            sample['upper_bound_1'] = 37
            sample['upper_bound_0'] = 37
            self.assertTrue(cqm.check_feasible(sample))

            # check infeasible configuration
            sample = original_sample.copy()
            sample[f'{v}_0'] = 35
            sample[f'z_0'] = 35
            sample['case_0_in_bin_0'] = 1
            sample['case_1_in_bin_0'] = 1
            sample['upper_bound_1'] = 30
            self.assertFalse(cqm.check_feasible(sample))

    def test_build_cqm(self):
        cqm, _ = build_cqm(self.variables, self.bins, self.cases)
        feasible_sample = {
            'o_0_0': 0.0, 'o_0_1': 0.0, 'o_0_2': 0.0, 'o_0_3': 0.0,
            'o_0_4': 0.0, 'o_0_5': 1.0, 'o_1_0': 0.0, 'o_1_1': 0.0,
            'o_1_2': 1.0, 'o_1_3': 0.0, 'o_1_4': 0.0, 'o_1_5': 0.0,
            'sel_0_1_0': 0.0, 'sel_0_1_1': 0.0, 'sel_0_1_2': 0.0,
            'sel_0_1_3': 0.0, 'sel_0_1_4': 1.0, 'sel_0_1_5': 0.0,
            'x_0': 0.0, 'x_1': 0.0, 'y_0': 3.0, 'y_1': 0.0,
            'z_0': 0.0, 'z_1': 0.0, 'upper_bound_0': 3.0}
        infeasible_sample = {
            'o_0_0': 0.0, 'o_0_1': 0.0, 'o_0_2': 0.0, 'o_0_3': 0.0,
            'o_0_4': 0.0, 'o_0_5': 1.0, 'o_1_0': 0.0, 'o_1_1': 0.0,
            'o_1_2': 0.0, 'o_1_3': 0.0, 'o_1_4': 1.0, 'o_1_5': 0.0,
            'sel_0_1_0': 0.0, 'sel_0_1_1': 0.0, 'sel_0_1_2': 1.0,
            'sel_0_1_3': 0.0, 'sel_0_1_4': 0.0, 'sel_0_1_5': 0.0,
            'x_0': 0.0, 'x_1': 0.0, 'y_0': 0.0, 'y_1': 0.0,
            'z_0': 0.0, 'z_1': 0.0, 'upper_bound_0': 3.0}

        self.assertTrue(cqm.check_feasible(feasible_sample))
        self.assertFalse(cqm.check_feasible(infeasible_sample))

    def test_call_solver(self):
        cqm, _ = build_cqm(self.variables, self.bins, self.cases)

        with patch('packing3d.LeapHybridCQMSampler') as mock:
            call_solver(cqm, time_limit=5, use_cqm_solver=True)
            mock.return_value.sample_cqm.assert_called_with(cqm, time_limit=5, label='3d bin packing')

        with patch.object(MIPCQMSolver, 'sample_cqm') as mock:
            call_solver(cqm, time_limit=5, use_cqm_solver=False)
            mock.assert_called_with(cqm, time_limit=5)
