from io import StringIO
import sys
import streamlit as st
from packing3d import (Cases,
                       Bins,
                       Variables,
                       build_cqm,
                       call_cqm_solver)
from utils import print_cqm_stats, plot_cuboids, read_instance

st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>3D Bin Packing Demo</h1>",
    unsafe_allow_html=True
)

problem_filepath = st.sidebar.text_input("Problem specification file",
                                         value="input/sample_data.txt")
bin_length = st.sidebar.number_input("Bin length", value=100)
bin_width = st.sidebar.number_input("Bin width", value=100)
bin_height = st.sidebar.number_input("Bin height", value=110)
num_bins = st.sidebar.number_input("Number of bins", value=1)
time_limit = st.sidebar.number_input("Hybrid solver time limit (S)", value=20)
run_button = st.sidebar.button("Run CQM Solver")

if run_button:
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

    st.plotly_chart(plotly_fig, use_container_width=True)


    def _get_cqm_stats() -> str:
        cqm_info_stream = StringIO()
        sys.stdout = cqm_info_stream
        print_cqm_stats(cqm)
        sys.stdout = sys.__stdout__

        return cqm_info_stream.getvalue()

    st.code(_get_cqm_stats())
else:
    st.text("Readme placeholder")
