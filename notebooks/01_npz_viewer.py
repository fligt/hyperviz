# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from fairdatanow import data_now

# %%
url = 'https://laboppad.nl/ukiyo-e-world' 

toml_txt = '''
# here are the 10 corresponding spectral data cubes processed by Gauthier and Tessa 
[data.npz] 
RV-1-4468-544 = ".*RIS/interim/.*RV-1-4468-544.*[.]npz"
RV-1-4469-484 = ".*RIS/interim/.*RV-1-4469-484.*[.]npz"
RV-1-4469-58 = ".*RIS/interim/.*RV-1-4469-58.*[.]npz"
RV-1-4469-x6 = ".*RIS/interim/.*RV-1-4469-x6.*[.]npz"
RV-1-4469Q = ".*RIS/interim/.*RV-1-4469Q.*[.]npz"
RV-1-4470-12 = ".*RIS/interim/.*RV-1-4470-12.*[.]npz"
RV-1-4470-27 = ".*RIS/interim/.*RV-1-4470-27.*[.]npz"
RV-360-2345g = ".*RIS/interim/.*RV-360-2345g.*[.]npz"
RV-360-2359-2 = ".*RIS/interim/.*RV-360-2359-2.*[.]npz"
RV-360-6886 = ".*RIS/interim/.*RV-360-6886.*[.]npz"
'''

data = data_now(url, toml_txt)

# %%
from dash import Dash, dcc, html, callback, Input, Output, State
import plotly.express as px
import numpy as np


# %%
def create_app(data: dict) -> Dash:

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Store(id='data_store', data=data),
        dcc.Dropdown(id="object_num_dropdown", options=list(data['npz'].keys()), placeholder='Select Object Number'),
        dcc.Graph(id="pseudo_rgb_graph", config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"], "scrollZoom":True})
    ])

    @app.callback(
        Output(component_id="pseudo_rgb_graph", component_property="figure"),
        Input(component_id="object_num_dropdown", component_property="value"),
        State(component_id="data_store", component_property="data"),
        prevent_initial_call=True
        )
    def on_dropdown_change(object_num_dropdown_value: str, data_store_data: dict) -> px.imshow:
        npz_file = data_store_data['npz'][object_num_dropdown_value][0]
        npz = np.load(npz_file)
        cube = npz['image'][:,:, ::-1].transpose(1, 2, 0)
        pseudo_rgb = cube[:,:, [70, 53, 19]] 

        pseudo_rgb_graph_figure = px.imshow(pseudo_rgb, binary_string=True)

        pseudo_rgb_graph_figure.update_layout(dragmode='drawrect')
        
        return pseudo_rgb_graph_figure

    return app

def make_dashboard(data: dict):
    app = create_app(data)
    app.run(debug=True)


# %%
make_dashboard(data)
