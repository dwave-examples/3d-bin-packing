from io import StringIO
import numpy as np
import sys
import streamlit as st
from packing3d import (Cases,
                       Bins,
                       Variables,
                       build_cqm,
                       call_cqm_solver)
from utils import print_cqm_stats, plot_cuboids, read_instance

def _get_cqm_stats(cqm) -> str:
    cqm_info_stream = StringIO()
    sys.stdout = cqm_info_stream
    print_cqm_stats(cqm)
    sys.stdout = sys.__stdout__

    return cqm_info_stream.getvalue()

st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>3D Bin Packing Demo</h1>",
    unsafe_allow_html=True
)

run_type = st.sidebar.radio(label="Choose run type:", 
                                   options=["Random","File upload"])

if run_type == "File upload":
    problem_filepath = st.sidebar.text_input(label="Problem instance file",
                                             value="input/sample_data.txt")
    time_limit = st.sidebar.number_input(label="Hybrid solver time limit (S)",
                                         value=20)
    display_input = st.sidebar.checkbox("Display input data")
    run_button = st.sidebar.button("Run CQM Solver")

    col1, col2 = st.columns([1,2])

    with col1:
        data = read_instance(problem_filepath)
        if display_input:
            st.write(data)

    with col2:
        if run_button:
            cases = Cases(data)
            bins = Bins(data, cases=cases)

            model_variables = Variables(cases, bins)

            cqm, origins = build_cqm(model_variables, bins, cases)

            best_feasible = call_cqm_solver(cqm, time_limit)

            plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                                    bins, origins)

            st.plotly_chart(plotly_fig, use_container_width=True)

            st.code(_get_cqm_stats(cqm))

elif run_type == "Random":
    col1, col2 = st.columns([1,2])
    with col1:
        with st.form(key="problem_config"):
            time_limit = st.number_input(label="Hybrid solver time limit(S)",
                                         value=20)
            num_bins = st.number_input("Number of bins", min_value=1,
                                        max_value=3)
            num_cases = st.number_input("Number of unique case types",
                                        min_value=1, max_value=5)
            bin_length = st.number_input("Bin length", min_value=50,
                                         max_value=100)
            bin_width = st.number_input("Bin width", min_value=50,
                                        max_value=100)
            bin_height = st.number_input("Bin height", min_value=50,
                                         max_value=100)
            form_submit = st.form_submit_button("Run random problem")

        if form_submit:
            rng = np.random.default_rng()

            bin_volume = num_bins * bin_length * bin_width * bin_height
            quantity = rng.integers(5,15,num_cases)
            max_case_side_length = np.floor((bin_volume/sum(quantity))**(1/3))

            data = {
                "num_bins":num_bins,
                "bin_dimensions":[bin_length, bin_width, bin_height],
                "quantity":quantity,
                "case_ids":np.array(range(num_cases)),
                "case_length":rng.integers(10,max_case_side_length,num_cases),
                "case_width":rng.integers(10,max_case_side_length,num_cases),
                "case_height":rng.integers(10,max_case_side_length,num_cases)
            }
        
            st.write(data)
        
            with col2:    
                cases = Cases(data)
                bins = Bins(data, cases=cases)

                model_variables = Variables(cases, bins)

                cqm, origins = build_cqm(model_variables, bins, cases)

                best_feasible = call_cqm_solver(cqm, time_limit)

                plotly_fig = plot_cuboids(best_feasible, model_variables,
                                          cases, bins, origins)

                st.plotly_chart(plotly_fig)
        
                st.code(_get_cqm_stats(cqm))