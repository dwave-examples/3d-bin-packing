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
    MIP = 1

    @property
    def label(self):
        return {
            SolverType.CQM: "Quantum Hybrid (CQM)",
            SolverType.MIP: "CBC (Python-MIP)",
        }[self]


class ProblemType(Enum):
    GENERATED = 0
    FILE = 1

    @property
    def label(self):
        return {
            ProblemType.GENERATED: "Generated",
            ProblemType.FILE: "Uploaded",
        }[self]
