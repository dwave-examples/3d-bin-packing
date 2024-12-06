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

"""This file stores input parameters for the app."""

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "3D Bin Packing"
MAIN_HEADER = "3D Bin Packing"
DESCRIPTION = """\
3D bin packing is an optimization problem where the goal is to use the minimum number of bins to
pack items with different dimensions, weights and properties. In this example, both items and bins
are cuboids, and the sides of the items must be packed parallel to the sides of bins.
"""

RANDOM_SEED = 4

TABLE_HEADERS = ["Case ID", "Quantity", "Length", "Width", "Height"]

#######################################
# Sliders, buttons and option entries #
#######################################

NUM_BINS = {
    "min": 1,
    "max": 5,
    "step": 1,
    "value": 1,
}

NUM_CASES = {
    "min": 1,
    "max": 75,
    "step": 1,
    "value": 20,
}

CASE_DIM = {
    "min": 1,
    "max": 30,
    "step": 1,
    "value": [2, 15],
}

BIN_DIM = {
    "min": 1,
    "max": 200,
    "step": 1,
    "value": 50,
}

ADVANCED_SETTINGS = ["Color by Case ID"]

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
