import numpy as np
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html


def create_simple_app(data: dict) -> Dash:

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Store(id="data_store", data=data),
            dcc.Dropdown(
                id="object_num_dropdown",
                options=list(data["npz"].keys()),
                placeholder="Select Object Number",
            ),
            dcc.Graph(
                id="pseudo_rgb_graph",
                config={
                    "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                    "scrollZoom": True,
                },
            ),
        ]
    )

    @app.callback(
        Output(component_id="pseudo_rgb_graph", component_property="figure"),
        Input(component_id="object_num_dropdown", component_property="value"),
        State(component_id="data_store", component_property="data"),
        prevent_initial_call=True,
    )
    def on_dropdown_change(
        object_num_dropdown_value: str, data_store_data: dict
    ) -> px.imshow:
        npz_file = data_store_data["npz"][object_num_dropdown_value][0]
        npz = np.load(npz_file)
        cube = npz["image"][:, :, ::-1].transpose(1, 2, 0)
        pseudo_rgb = cube[:, :, [70, 53, 19]]

        pseudo_rgb_graph_figure = px.imshow(pseudo_rgb, binary_string=True)

        pseudo_rgb_graph_figure.update_layout(dragmode="drawrect")

        return pseudo_rgb_graph_figure

    return app


def make_viewer(data: dict) -> None:
    """Launch an interactive viewer for hyperspectral NPZ data.

    Parameters
    ----------
    data
        Mapping with an ``npz`` entry. Each key below ``npz`` identifies an
        object and maps to a sequence whose first item is the path to its NPZ
        file. Each NPZ file must contain an ``image`` array.

    Notes
    -----
    This function starts a Dash development server and blocks until that
    server is stopped.
    """
    app = create_simple_app(data)
    app.run(debug=True)
