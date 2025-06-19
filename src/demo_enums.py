# Copyright 2024 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class SolverType(Enum):
    CQM = 0
    HIGHS = 1

    @property
    def label(self):
        return {
            SolverType.CQM: "Quantum Hybrid (CQM)",
            SolverType.HIGHS: "Classical (HiGHS)",
        }[self]


class ProblemType(Enum):
    GENERATED = 0
    FILE = 1
    SCENARIO = 2

    @property
    def label(self):
        return {
            ProblemType.GENERATED: "Generated",
            ProblemType.FILE: "Uploaded",
            ProblemType.SCENARIO: "Default Scenarios",
        }[self]


class ScenarioType(Enum):
    ONE_FEW = 0
    ONE_MANY = 1
    TWO_FEW = 2
    TWO_MANY = 3

    @property
    def label(self):
        return {
            ScenarioType.ONE_FEW: "One Bin - Few Cases",
            ScenarioType.ONE_MANY: "One Bin - Many Cases",
            ScenarioType.TWO_FEW: "Two Bins - Few Cases",
            ScenarioType.TWO_MANY: "Two Bins - Many Cases",
        }[self]
