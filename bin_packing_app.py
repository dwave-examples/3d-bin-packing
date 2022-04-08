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
    run_button = st.sidebar.button("Run CQM Solver")

    if run_button:
        col1, col2 = st.columns(2)
        with col1:
            data = read_instance(problem_filepath)
            cases = Cases(data)
            bins = Bins(data, cases=cases)

            model_variables = Variables(cases, bins)

            cqm, origins = build_cqm(model_variables, bins, cases)

            best_feasible = call_cqm_solver(cqm, time_limit)

            # kwargs for ploty 3D meshes
            plot_kwargs = dict(alphahull=0, flatshading=True, showlegend=True)

            plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                                    bins, origins, **plot_kwargs)

            st.plotly_chart(plotly_fig)

        with col2:
            st.code(_get_cqm_stats(cqm))
elif run_type == "Random":
    col1, col2 = st.columns([1,2])
    with col1:
        with st.form(key="problem_config"):
            time_limit = st.number_input(label="Hybrid solver time limit(S)",
                                         value=20)
            num_bins = st.slider("Number of bins", 1, 3)
            num_cases = st.slider("Number of unique case dimensions", 1, 5)
            form_submit = st.form_submit_button("Run random problem")
        
    with col2:    
        if form_submit:
            rng = np.random.default_rng()
            data = {
                "num_bins":num_bins,
                "bin_dimensions":[100,100,100],
                "quantity":rng.integers(8,12,num_cases),
                "case_ids":[range(num_cases)],
                "case_length":rng.integers(15,25,num_cases),
                "case_width":rng.integers(15,25,num_cases),
                "case_height":rng.integers(15,25,num_cases)
            }
            cases = Cases(data)
            bins = Bins(data, cases=cases)

            model_variables = Variables(cases, bins)

            cqm, origins = build_cqm(model_variables, bins, cases)

            best_feasible = call_cqm_solver(cqm, time_limit)

            # kwargs for ploty 3D meshes
            plot_kwargs = dict(alphahull=0, flatshading=True, showlegend=True)

            plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                                    bins, origins, **plot_kwargs)

            st.plotly_chart(plotly_fig)
    
            st.code(_get_cqm_stats(cqm))