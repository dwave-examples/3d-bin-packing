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

from packing3d import Cases, Bins

from utils import read_instance, write_input_data


class TestUtils(unittest.TestCase):

    def test_read_write_cqm(self):
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

