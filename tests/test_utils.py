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

from io import StringIO
import dimod
import os
import sys
from tempfile import TemporaryDirectory
import unittest

from utils import (print_cqm_stats,
                   plot_cuboids,
                   read_instance, 
                   write_input_data, 
                   write_solution_to_file)

from packing3d import (Cases,
                       Bins,
                       Variables,
                       build_cqm,
                       call_solver)


class TestUtils(unittest.TestCase):
    def test_print_cqm_stats(self):
        test_bqm = dimod.BQM('BINARY')
        with self.assertRaises(ValueError):
            print_cqm_stats(test_bqm)

        test_cqm = dimod.CQM()
        r = dimod.Real('r')
        b = dimod.Binary('b')
        i = dimod.Integer('i')
        test_cqm.add_constraint(r == 1)
        test_cqm.add_constraint(b >= 0.5)
        test_cqm.add_constraint(i <= 2)

        cqm_stats_stream = StringIO()
        sys.stdout = cqm_stats_stream
        print_cqm_stats(test_cqm)
        sys.stdout = sys.__stdout__
        printed_string = cqm_stats_stream.getvalue()

        expected_field_values = ['1','1','1','0','3','0','1','1','1']

        self.assertEqual(printed_string.splitlines()[-1].split(),
                         expected_field_values)

    def test_plot_cuboids(self):
        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases=cases)
        variables = Variables(cases, bins)
        cqm, effective_dimensions = build_cqm(variables, bins, cases)

        best_feasible = call_solver(cqm, time_limit=3, use_cqm_solver=False)

        fig = plot_cuboids(best_feasible, variables, cases, bins, 
                           effective_dimensions)
        
        expected_num_cases = 2
        expected_num_boundaries = 3
        self.assertEqual(
            len(fig['data']), expected_num_cases + expected_num_boundaries
        )

    def test_read_write_input_data(self):
        data = read_instance(instance_path='./tests/test_data_1.txt')
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

    def test_write_solution_to_file(self):
        data = read_instance(instance_path='./tests/test_data_1.txt')
        cases = Cases(data)
        bins = Bins(data, cases=cases)
        variables = Variables(cases, bins)
        cqm, effective_dimensions = build_cqm(variables, bins, cases)

        best_feasible = call_solver(cqm, time_limit=3, use_cqm_solver=False)

        with TemporaryDirectory() as tempdir:
            solution_file_path = os.path.join(tempdir, "solution_test.txt")
            write_solution_to_file(solution_file_path,
                                   cqm,
                                   variables,
                                   best_feasible,
                                   cases,
                                   bins,
                                   effective_dimensions)
            self.assertTrue(os.path.exists(solution_file_path))
            