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
import numpy as np
import sys
import streamlit as st
from typing import Optional
from packing3d import (Cases,
                       Bins,
                       Variables,
                       build_cqm,
                       call_solver)
from utils import (print_cqm_stats,
                   plot_cuboids,
                   read_instance,
                   write_solution_to_file,
                   write_input_data)


def _get_cqm_stats(cqm) -> str:
    cqm_info_stream = StringIO()
    sys.stdout = cqm_info_stream
    print_cqm_stats(cqm)
    sys.stdout = sys.__stdout__

    return cqm_info_stream.getvalue()


def _solve_bin_packing_instance(data: dict,
                                write_to_file: bool,
                                solution_filename: Optional[str],
                                use_cqm_solver: bool = True,
                                **st_plotly_kwargs):
    cases = Cases(data)
    bins = Bins(data, cases=cases)

    model_variables = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(model_variables, bins, cases)

    best_feasible = call_solver(cqm, time_limit, use_cqm_solver)

    plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                              bins, effective_dimensions, color_coded)

    st.plotly_chart(plotly_fig, **st_plotly_kwargs)

    st.code(_get_cqm_stats(cqm))

    if write_to_file:
        write_solution_to_file(solution_filename, cqm, 
                               model_variables, best_feasible,
                               cases, bins, effective_dimensions)


st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>3D Bin Packing Demo</h1>",
    unsafe_allow_html=True
)

run_type = st.sidebar.radio(label="Choose run type:",
                            options=["Random", "File upload"])

solver_type = st.sidebar.radio(label="Choose solver to run problems on:",
                               options=["Constrained Quadratic Model",
                                        "CBC (Python-MIP)",
                                        ])

if solver_type == "Constrained Quadratic Model":
    use_cqm_solver = True
else:
    use_cqm_solver = False

if run_type == "File upload":
    problem_filepath = st.sidebar.text_input(label="Problem instance file",
                                             value="input/sample_data_1.txt")
    time_limit = st.sidebar.number_input(label="Hybrid solver time limit (S)",
                                         value=20)
    color_coded = st.sidebar.checkbox("Color coded cases")
    display_input = st.sidebar.checkbox("Display input data")
    write_to_file = st.sidebar.checkbox("Write solution to file")
    if write_to_file:
        solution_filename = st.sidebar.text_input("Solution filename")
    else:
        solution_filename = None
    run_button = st.sidebar.button("Run")

    if display_input:
        col1, col2 = st.columns([1, 2])
        with col1:
            if problem_filepath:
                with open(problem_filepath) as f:
                    for line in f:
                        st.text(line)

        with col2:
            if run_button:
                data = read_instance(problem_filepath)
                _solve_bin_packing_instance(data,
                                            write_to_file,
                                            solution_filename,
                                            use_cqm_solver,
                                            **{"use_container_width": True})
    else:
        if run_button:
            data = read_instance(problem_filepath)
            _solve_bin_packing_instance(data,
                                        write_to_file,
                                        solution_filename,
                                        use_cqm_solver,
                                        **{"use_container_width": True})


elif run_type == "Random":
    color_coded = st.sidebar.checkbox("Color coded cases")
    display_input = st.sidebar.checkbox("Display input data")
    random_seed = st.sidebar.checkbox("Set random seed")
    if random_seed:
        seed = st.sidebar.number_input("Random seed", value=0)

    save_input_to_file = st.sidebar.checkbox("Save input data to file")
    if save_input_to_file:
        input_filename = st.sidebar.text_input("input filename")
    else:
        input_filename = None

    write_to_file = st.sidebar.checkbox("Write solution to file")
    if write_to_file:
        solution_filename = st.sidebar.text_input("Solution filename")
    else:
        solution_filename = None

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form(key="problem_config"):
            time_limit = st.number_input(label="Hybrid solver time limit(S)",
                                         value=20)
            num_bins = st.number_input("Number of bins", min_value=1,
                                       max_value=5)
            num_cases = st.number_input("Number of cases",
                                        min_value=1, max_value=75, value=20)
            case_size_range = st.slider("Case dimension range", min_value=1,
                                        max_value=30, value=(1, 15))
            bin_length = st.number_input("Bin length", min_value=1,
                                         max_value=200, value=50)
            bin_width = st.number_input("Bin width", min_value=1,
                                        max_value=200, value=50)
            bin_height = st.number_input("Bin height", min_value=1,
                                         max_value=200, value=50)
            form_submit = st.form_submit_button("Run")

        if form_submit:
            if random_seed:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()

            data = {
                "num_bins": num_bins,
                "bin_dimensions": [bin_length, bin_width, bin_height],
                "case_length": rng.integers(
                    case_size_range[0], case_size_range[1], 
                    num_cases, endpoint=True
                ),
                "case_width": rng.integers(
                    case_size_range[0], case_size_range[1], 
                    num_cases, endpoint=True
                ),
                "case_height": rng.integers(
                    case_size_range[0], case_size_range[1], 
                    num_cases, endpoint=True
                ),
            }

            # Determine quantities and case_ids
            case_dimensions = np.vstack(
                [data["case_length"], data["case_width"], data["case_height"]]
            )
            unique_dimensions, data["quantity"] = np.unique(case_dimensions, 
                                                            axis=1,
                                                            return_counts=True)
            
            data["case_length"] = unique_dimensions[0,:]
            data["case_width"] = unique_dimensions[1,:]
            data["case_height"] = unique_dimensions[2,:]
            
            data["case_ids"] = np.array(range(len(data["quantity"])))

            input_data_string = write_input_data(data, input_filename)
            if display_input:
                for line in input_data_string.split(sep='\n'):
                    st.text(line)

            with col2:
                _solve_bin_packing_instance(data,
                                            write_to_file,
                                            solution_filename,
                                            use_cqm_solver,
                                            **{"use_container_width": True})
