from io import StringIO
from matplotlib.pyplot import plot
import sys
import streamlit as st
from packing3d import (Cases,
                         Pallets,
                         Variables,
                         build_cqm,
                         call_cqm_solver)
from utils import print_cqm_stats, plot_cuboids

st.header("3D Bin Packing Demo")

problem_filepath = st.sidebar.text_input("Problem specification file",
                                         value="input/sample_data.txt")
pallet_length = st.sidebar.number_input("Pallet length", value=100)
pallet_width = st.sidebar.number_input("Pallet width", value=100)
pallet_height = st.sidebar.number_input("Pallet height",value=110)
num_pallets = st.sidebar.number_input("Number of pallets",value=1)
time_limit = st.sidebar.number_input("Hybrid solver time limit (S)",value=20)

cases = Cases(problem_filepath)
pallets = Pallets(length=pallet_length,
                  width=pallet_width,
                  height=pallet_height,
                  num_pallets=num_pallets,
                  cases=cases)

model_variables = Variables(cases, pallets)

cqm, origins = build_cqm(model_variables, pallets, cases)

best_feasible = call_cqm_solver(cqm, time_limit)

#kwargs for ploty 3D meshes
plot_kwargs = dict(alphahull=0, opacity=0.75, flatshading=True)

plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                          pallets, origins, **plot_kwargs)

st.plotly_chart(plotly_fig, use_container_width=True)

def _get_cqm_stats() -> str:
    cqm_info_stream = StringIO()
    sys.stdout = cqm_info_stream
    print_cqm_stats(cqm)
    sys.stdout = sys.__stdout__

    return cqm_info_stream.getvalue()

st.code(_get_cqm_stats())
