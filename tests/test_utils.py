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

import os
import sys
import unittest
from io import StringIO
from tempfile import TemporaryDirectory

import dimod

from packing3d import Bins, Cases, Variables, build_cqm, call_solver
from utils import (
    case_list_to_dict,
    plot_cuboids,
    print_cqm_stats,
    read_instance,
    write_input_data,
    write_solution_to_file,
)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestUtils(unittest.TestCase):
    def test_print_cqm_stats(self):
        test_bqm = dimod.BQM("BINARY")
        with self.assertRaises(ValueError):
            print_cqm_stats(test_bqm)

        test_cqm = dimod.CQM()
        r = dimod.Real("r")
        b = dimod.Binary("b")
        i = dimod.Integer("i")
        test_cqm.add_constraint(r == 1)
        test_cqm.add_constraint(b >= 0.5)
        test_cqm.add_constraint(i <= 2)

        cqm_stats_stream = StringIO()
        sys.stdout = cqm_stats_stream
        print_cqm_stats(test_cqm)
        sys.stdout = sys.__stdout__
        printed_string = cqm_stats_stream.getvalue()

        expected_field_values = ["1", "1", "1", "0", "3", "0", "1", "1", "1"]

        self.assertEqual(printed_string.splitlines()[-1].split(), expected_field_values)

    def test_plot_cuboids(self):
        data = read_instance(instance_path=project_dir + "/tests/test_data_1.txt")
        cases = Cases(data)
        bins = Bins(data, cases=cases)
        variables = Variables(cases, bins)
        cqm, effective_dimensions = build_cqm(variables, bins, cases)

        best_feasible = call_solver(cqm, time_limit=3, use_cqm_solver=False)

        fig = plot_cuboids(best_feasible, variables, cases, bins, effective_dimensions)

        expected_num_cases = 2
        expected_num_boundaries = 3
        self.assertEqual(len(fig["data"]), expected_num_cases + expected_num_boundaries)

    def test_read_write_input_data(self):
        data = read_instance(instance_path=project_dir + "/tests/test_data_1.txt")
        out_file_string = write_input_data(data)

        out_list = out_file_string.split(sep="\n")
        case_info = []
        for i, line in enumerate(out_list):
            if i == 0:
                num_bins = int(line.split()[-1])
            elif i == 1:
                bin_dimensions = [int(i) for i in line.split()[-3:]]
            elif 2 <= i <= 4:
                continue
            else:
                case_info.append(list(map(int, line.split())))

        data1 = case_list_to_dict(case_info, num_bins, bin_dimensions)

        self.assertEqual(
            data1,
            {
                "Case ID": [0, 1],
                "Quantity": [1, 1],
                "Length": [2, 3],
                "Width": [2, 3],
                "Height": [2, 3],
                "num_bins": 1,
                "bin_dimensions": [30, 40, 50],
            },
        )
        self.assertEqual(data1, data)

    def test_write_solution_to_file(self):
        data = read_instance(instance_path=project_dir + "/tests/test_data_1.txt")
        cases = Cases(data)
        bins = Bins(data, cases=cases)
        variables = Variables(cases, bins)
        cqm, effective_dimensions = build_cqm(variables, bins, cases)

        best_feasible = call_solver(cqm, time_limit=3, use_cqm_solver=False)

        with TemporaryDirectory() as tempdir:
            solution_file_path = os.path.join(tempdir, "solution_test.txt")
            write_solution_to_file(
                solution_file_path, cqm, variables, best_feasible, cases, bins, effective_dimensions
            )
            self.assertTrue(os.path.exists(solution_file_path))
