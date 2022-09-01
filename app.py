from dash import dcc, html

from components import build_banner
from components import build_tabs
from components import app


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=1000,
            n_intervals=50,
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                html.Div(id="app-content"),
            ],
        ),
    ],
)


if __name__ == "__main__":
    app.run_server(debug=True, port=8501)
