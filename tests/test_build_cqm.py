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
from packing3d import build_cqm, Cases, Bins, Variables
from packing3d import _add_orientation_constraints, _add_bin_on_constraint
from packing3d import _add_boundary_constraints, _add_geometric_constraints

from utils import read_instance, write_input_data


class TestBuildCQM(unittest.TestCase):
    def test_add_orientation_constraints(self):
        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        vars = Variables(cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        _add_orientation_constraints(cqm, vars, cases)
        self.assertEqual(len(cqm.constraints), 2)
        for c, cval in cqm.constraints.items():
            self.assertEqual(cval.lhs.num_variables, 6)
            self.assertEqual(sum((cval.lhs.linear.values())), 6)
            self.assertEqual(cval.sense, dimod.sym.Sense.Eq)

    def test_add_bin_on_constraint(self):
        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        bins.num_bins = 2
        vars = Variables(cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        _add_bin_on_constraint(cqm, vars, bins, cases)

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

        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        vars = Variables(cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        effective_dimensions = _add_orientation_constraints(cqm, vars, cases)
        _add_geometric_constraints(cqm, vars, bins, cases, effective_dimensions)
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
        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        bins.num_bins = 2
        vars = Variables(cases, bins)
        cqm = dimod.ConstrainedQuadraticModel()
        effective_dimensions = _add_orientation_constraints(cqm, vars, cases)
        _add_boundary_constraints(cqm, vars, bins, cases, effective_dimensions)
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
        try:
            data = read_instance(instance_path='test_data_1.txt')
        except:
            data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        vars = Variables(cases, bins)
        cqm, _ = build_cqm(vars, bins, cases)
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

    def test_read_write_cqm(self):
        try:
            data = read_instance(instance_path='test_data_1.txt')
        except:
            data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases)
        out_file_string = write_input_data(data)
        data1 = {"num_bins": 0, "bin_dimensions": [], "quantity": [],
                 "case_ids": [], "case_length": [], "case_width": [],
                 "case_height": []}
        out_list = (out_file_string.split(sep='\n'))
        for i, line in enumerate(out_list):
            if i == 0:
                data1["num_bins"] = int(line.split()[-1])
            elif i == 1:
                data1["bin_dimensions"] = [int(i) for i in line.split()[-3:]]
            elif 2 <= i <= 4:
                continue
            else:
                case_info = list(map(int, line.split()))
                data1["case_ids"].append(case_info[0])
                data1["quantity"].append(case_info[1])
                data1["case_length"].append(case_info[2])
                data1["case_width"].append(case_info[3])
                data1["case_height"].append(case_info[4])

        self.assertEqual(data1, {'num_bins': 1, 'bin_dimensions': [30, 40, 50],
                                 'quantity': [1, 1], 'case_ids': [0, 1],
                                 'case_length': [2, 3], 'case_width': [2, 3],
                                 'case_height': [2, 3]})
        self.assertEqual(data1, data)


if __name__ == '__main__':
    TestBuildCQM()
