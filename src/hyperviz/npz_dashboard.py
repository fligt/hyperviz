import base64
import re
import uuid

import dash_daq as daq
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tomlkit
from dash import (
    Dash,
    Input,
    Output,
    Patch,
    State,
    ctx,
    dcc,
    html,
    no_update,
)
from PIL import Image

#CONSTANTS
DEFAULT_COLOR = "#119DFF"
ROI_COORDINATES = ("x0", "x1", "y0", "y1")
SHAPE_COORDINATE = re.compile(r"^shapes\[(\d+)]\.(x0|x1|y0|y1)$")
VIEW_SIZE = (900, 700)


def _load_cube(data_dict: dict, object_num: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads the cube and wavelengths from the npz file using np.load()."""
    with np.load(data_dict["npz"][object_num][0]) as npz:
        cube = npz["image"][:, :, ::-1].transpose(1, 2, 0)
        wavelengths = npz["wavelengths"]
    return cube, wavelengths


def _load_tif(data_dict: dict, object_num: str) -> Image.Image:
    """Loads the TIF image using PIL"""
    with Image.open(data_dict["tif"][object_num][0]) as source:
        return source.convert("RGB")


def _image_figure(
    image: Image.Image,
    x_range: list[float] | tuple[float, float] | None = None,
    y_range: list[float] | tuple[float, float] | None = None,
) -> go.Figure:
    """Apply the LANCZOS algorithm to handle big TIF files in the viewer."""
    width, height = image.size
    x0, x1 = sorted(np.clip(x_range or (0, width), 0, width))
    y0, y1 = sorted(np.clip(y_range or (0, height), 0, height))
    x0, x1 = int(np.floor(x0)), int(np.ceil(x1))
    y0, y1 = int(np.floor(y0)), int(np.ceil(y1))
    x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)

    rendered = image.crop((x0, y0, x1, y1))
    rendered.thumbnail(VIEW_SIZE, Image.Resampling.LANCZOS)
    figure = go.Figure()
    figure.add_layout_image(
        source=rendered,
        xref="x",
        yref="y",
        x=x0,
        y=y0,
        sizex=x1 - x0,
        sizey=y1 - y0,
        xanchor="left",
        yanchor="top",
        sizing="stretch",
        layer="below",
    )
    figure.update_layout(
        dragmode="pan",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"range": [x0, x1], "visible": False},
        yaxis={"range": [y1, y0], "visible": False, "scaleanchor": "x"},
    )
    return figure


def _normalize_roi(
    shape: dict,
    name: str,
    color: str,
    roi_id: str | None = None,
) -> dict:
    """Turns a shape into a readable ROI format."""
    return {
        "id": roi_id or uuid.uuid4().hex,
        "name": name,
        "color": color,
        **{key: float(shape[key]) for key in ROI_COORDINATES},
    }


def _unique_roi_name(rois: list[dict]) -> str:
    """Handles the naming of non-labeled ROIs."""
    names = {roi["name"] for roi in rois}
    number = 1
    while f"ROI {number}" in names:
        number += 1
    return f"ROI {number}"


def _rois_from_toml(content: str) -> dict[str, list[dict]]:
    """Gets the ROIs from the uploaded TOML document."""
    document = tomlkit.parse(content)
    return {
        object_num: [
            _normalize_roi(roi, name, roi.get("color", DEFAULT_COLOR))
            for name, roi in object_rois.items()
        ]
        for object_num, object_rois in document.get("roi", {}).items()
    }


def _graph(graph_id: str, draw: bool = False) -> dcc.Graph:
    """Adds styling and drawing capabilities to the graphs."""
    config: dict[str, bool | list[str]] = {
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    }
    if draw:
        config["modeBarButtonsToAdd"] = ["drawrect", "eraseshape"]
    return dcc.Graph(
        id=graph_id,
        figure={},
        config=config,
        style={"height": "700px", "width": "100%"},
    )


def create_dashboard(
    data_dict: dict,
    toml_text: str,
    *,
    default_object: str | None = None,
    title: str = "Spectral Image Dashboard",
) -> Dash:
    """Create a spectral-image dashboard without starting its server.

    Parameters
    ----------
    data_dict
        Mapping with matching ``npz`` and ``tif`` entries for each object.
        NPZ files must contain ``image`` and ``wavelengths`` arrays.
    toml_text
        TOML configuration retained by the dashboard when ROI annotations are
        downloaded.
    default_object
        Object selected when the dashboard opens. The first NPZ object is used
        when this is omitted.
    title
        Heading displayed above the dashboard.

    Returns
    -------
    Dash
        The configured Dash application. Call ``run`` on it to start a server.

    Raises
    ------
    ValueError
        If no NPZ objects are supplied or ``default_object`` is unknown.
    """
    object_numbers = list(data_dict["npz"])
    if not object_numbers:
        raise ValueError("data_dict must contain at least one NPZ object")
    if default_object is None:
        default_object = object_numbers[0]
    if default_object not in data_dict["npz"]:
        raise ValueError(f"Unknown default object: {default_object}")
    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Store(id="roi-store", data={}),
            dcc.Store(id="toml-store", data=toml_text),
            dcc.Store(id="pseudo-size"),
            html.H1(title),
            dcc.Dropdown(
                list(data_dict["npz"]),
                value=default_object,
                clearable=False,
                id="object",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H3("TIFF"), _graph("tif")],
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [html.H3("Pseudo RGB"), _graph("pseudo", draw=True)],
                        style={"minWidth": 0},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "8px",
                },
            ),
            html.Div(
                [
                    daq.ColorPicker(
                        id="color", label="ROI color", value={"hex": DEFAULT_COLOR}
                    ),
                    dcc.Input(id="roi-name", placeholder="ROI name", type="text"),
                    dcc.Upload(id="upload", children=html.Button("Upload ROI TOML")),
                    html.Button("Save ROIs", id="save"),
                    html.Span(id="upload-status"),
                    dcc.Download(id="download"),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "center"},
            ),
            dcc.Graph(id="spectrum"),
        ]
    )

    def _roi_bounds(roi: dict, width: int, height: int) -> tuple[int, int, int, int]:
        x0, x1 = sorted(max(0, min(width, int(roi[key]))) for key in ("x0", "x1"))
        y0, y1 = sorted(max(0, min(height, int(roi[key]))) for key in ("y0", "y1"))
        return x0, x1, y0, y1

    def _roi_shapes(
        rois: list[dict], width: int, height: int
    ) -> list[dict[str, object]]:
        shapes: list[dict[str, object]] = []
        for roi in rois:
            x0, x1, y0, y1 = _roi_bounds(roi, width, height)
            shapes.append(
                {
                    "type": "rect",
                    "name": roi["id"],
                    "editable": True,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "line": {"color": roi["color"], "width": 4},
                }
            )
        return shapes

    def _spectrum_figure(
        cube: np.ndarray,
        wavelengths: np.ndarray,
        rois: list[dict],
    ) -> go.Figure:
        height, width, _ = cube.shape
        spectrum = go.Figure(
            go.Scatter(
                x=wavelengths,
                y=cube.mean(axis=(0, 1)),
                mode="lines",
                name="Full image",
            )
        )
        for roi in rois:
            x0, x1, y0, y1 = _roi_bounds(roi, width, height)
            roi_cube = cube[y0:y1, x0:x1]
            if roi_cube.size:
                spectrum.add_scatter(
                    x=wavelengths,
                    y=roi_cube.mean(axis=(0, 1)),
                    mode="lines",
                    name=roi["name"],
                    line={"color": roi["color"]},
                )
        spectrum.update_layout(
            title="Mean spectra",
            xaxis_title="Wavelength",
            yaxis_title="Intensity",
        )
        return spectrum

    @app.callback(
        Output("pseudo", "figure"),
        Output("spectrum", "figure"),
        Output("pseudo-size", "data"),
        Input("object", "value"),
        State("roi-store", "data"),
        State("color", "value"),
    )
    def _render_object(
        object_num: str,
        roi_store: dict | None,
        color: dict[str, str] | None,
    ) -> tuple[go.Figure, go.Figure, list[int]]:
        cube, wavelengths = _load_cube(data_dict, object_num)
        height, width, _ = cube.shape
        rois = (roi_store or {}).get(object_num, [])
        pseudo = px.imshow(cube[:, :, [70, 53, 19]], binary_string=True)
        pseudo.update_layout(
            dragmode="drawrect",
            newshape={
                "line": {"color": (color or {}).get("hex", DEFAULT_COLOR), "width": 4}
            },
            shapes=_roi_shapes(rois, width, height),
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"visible": False},
            yaxis={"visible": False, "scaleanchor": "x"},
        )
        return pseudo, _spectrum_figure(cube, wavelengths, rois), [width, height]

    @app.callback(
        Output("pseudo", "figure", allow_duplicate=True),
        Output("spectrum", "figure", allow_duplicate=True),
        Input("roi-store", "data"),
        State("object", "value"),
        prevent_initial_call=True,
    )
    def _render_rois(
        roi_store: dict | None, object_num: str
    ) -> tuple[Patch, go.Figure]:
        cube, wavelengths = _load_cube(data_dict, object_num)
        height, width, _ = cube.shape
        rois = (roi_store or {}).get(object_num, [])
        patch = Patch()
        patch["layout"]["shapes"] = _roi_shapes(rois, width, height)
        return patch, _spectrum_figure(cube, wavelengths, rois)

    @app.callback(
        Output("pseudo", "figure", allow_duplicate=True),
        Input("color", "value"),
        prevent_initial_call=True,
    )
    def _update_drawing_color(color: dict[str, str] | None) -> Patch:
        patch = Patch()
        patch["layout"]["newshape"]["line"]["color"] = (color or {}).get(
            "hex", DEFAULT_COLOR
        )
        return patch

    @app.callback(Output("tif", "figure"), Input("object", "value"))
    def _render_tif(object_num: str) -> go.Figure:
        return _image_figure(_load_tif(data_dict, object_num))

    @app.callback(
        Output("roi-store", "data"),
        Input("pseudo", "relayoutData"),
        State("object", "value"),
        State("roi-store", "data"),
        State("color", "value"),
        State("roi-name", "value"),
        prevent_initial_call=True,
    )
    def _update_rois(
        event: dict | None,
        object_num: str,
        roi_store: dict | None,
        color: dict[str, str] | None,
        requested_name: str | None,
    ) -> dict | object:
        if not event:
            return no_update
        current = list((roi_store or {}).get(object_num, []))
        updated = [dict(roi) for roi in current]

        if "shapes" in event:
            existing = {roi["id"]: roi for roi in current}
            updated = []
            for shape in event["shapes"]:
                previous = existing.get(shape.get("name"))
                updated.append(
                    _normalize_roi(
                        shape,
                        previous["name"]
                        if previous
                        else requested_name or _unique_roi_name(current + updated),
                        previous["color"]
                        if previous
                        else (color or {}).get("hex", DEFAULT_COLOR),
                        previous["id"] if previous else None,
                    )
                )
        else:
            changed = False
            for key, value in event.items():
                match = SHAPE_COORDINATE.match(key)
                if match and int(match.group(1)) < len(updated):
                    updated[int(match.group(1))][match.group(2)] = float(value)
                    changed = True
            if not changed:
                return no_update

        result = dict(roi_store or {})
        result[object_num] = updated
        return result

    def _visible_range(
        event: dict, width: int, height: int
    ) -> tuple[list[float] | None, list[float] | None]:
        if event.get("xaxis.autorange") or event.get("yaxis.autorange"):
            return [0, width], [height, 0]
        x_range = event.get("xaxis.range") or [
            event.get("xaxis.range[0]"),
            event.get("xaxis.range[1]"),
        ]
        y_range = event.get("yaxis.range") or [
            event.get("yaxis.range[0]"),
            event.get("yaxis.range[1]"),
        ]
        return (x_range, y_range) if None not in x_range + y_range else (None, None)

    @app.callback(
        Output("pseudo", "figure", allow_duplicate=True),
        Output("tif", "figure", allow_duplicate=True),
        Input("pseudo", "relayoutData"),
        Input("tif", "relayoutData"),
        State("object", "value"),
        State("pseudo-size", "data"),
        prevent_initial_call=True,
    )
    def _sync_views(
        pseudo_event: dict | None,
        tif_event: dict | None,
        object_num: str,
        pseudo_size: list[int] | None,
    ) -> tuple[object, object]:
        if not pseudo_size:
            return no_update, no_update
        pw, ph = pseudo_size
        from_pseudo = ctx.triggered_id == "pseudo"
        event = pseudo_event if from_pseudo else tif_event

        if from_pseudo:
            x_range, y_range = _visible_range(event or {}, pw, ph)
            if x_range is None:
                return no_update, no_update

        tif = _load_tif(data_dict, object_num)
        tw, th = tif.size
        if from_pseudo:
            tif_x = [x * tw / pw for x in x_range]
            tif_y = [y * th / ph for y in y_range]
        else:
            tif_x, tif_y = _visible_range(event or {}, tw, th)
            x_range, y_range = tif_x, tif_y
        if x_range is None:
            return no_update, no_update

        rendered = _image_figure(tif, tif_x, tif_y).layout.images[0]

        pseudo_patch, tif_patch = Patch(), Patch()
        for key in ("source", "x", "y", "sizex", "sizey"):
            tif_patch["layout"]["images"][0][key] = getattr(rendered, key)
        if from_pseudo:
            tif_patch["layout"]["xaxis"]["range"] = tif_x
            tif_patch["layout"]["yaxis"]["range"] = tif_y
        else:
            pseudo_patch["layout"]["xaxis"]["range"] = [x * pw / tw for x in x_range]
            pseudo_patch["layout"]["yaxis"]["range"] = [y * ph / th for y in y_range]
        return pseudo_patch, tif_patch

    @app.callback(
        Output("roi-store", "data", allow_duplicate=True),
        Output("toml-store", "data"),
        Output("upload-status", "children"),
        Input("upload", "contents"),
        prevent_initial_call=True,
    )
    def _upload_toml(contents: str | None) -> tuple[object, object, object]:
        if not contents:
            return no_update, no_update, no_update
        try:
            _, encoded = contents.split(",", 1)
            text = base64.b64decode(encoded, validate=True).decode()
            return _rois_from_toml(text), text, "ROI TOML loaded"
        except (
            ValueError,
            tomlkit.exceptions.ParseError,
        ):
            return no_update, no_update, "Could not load ROI TOML"

    @app.callback(
        Output("download", "data"),
        Input("save", "n_clicks"),
        State("roi-store", "data"),
        State("toml-store", "data"),
        prevent_initial_call=True,
    )
    def _download_toml(
        _clicks: int | None, roi_store: dict | None, base_toml: str
    ) -> dict[str, str]:
        document = tomlkit.parse(base_toml)
        roi_table = tomlkit.table()
        for object_num, rois in (roi_store or {}).items():
            object_table = tomlkit.table()
            for roi in rois:
                object_table[roi["name"]] = {
                    key: roi[key] for key in ("color", *ROI_COORDINATES)
                }
            roi_table[object_num] = object_table
        document["roi"] = roi_table
        return {"content": tomlkit.dumps(document), "filename": "question.toml"}

    return app


def make_dashboard(
    data_dict: dict,
    toml_text: str,
    *,
    default_object: str | None = None,
    title: str = "Spectral Image Dashboard",
) -> None:
    """Create and launch an interactive spectral-image dashboard.

    Parameters
    ----------
    data_dict
        Mapping with matching ``npz`` and ``tif`` entries for each object.
        NPZ files must contain ``image`` and ``wavelengths`` arrays.
    toml_text
        TOML configuration retained by the dashboard when ROI annotations are
        downloaded.
    default_object
        Object selected when the dashboard opens. The first NPZ object is used
        when this is omitted.
    title
        Heading displayed above the dashboard.

    Notes
    -----
    This function starts Dash in external Jupyter mode and blocks until the
    server is stopped. Use ``create_dashboard`` when the application needs
    to be configured or embedded before it is run.
    """
    app = create_dashboard(
        data_dict,
        toml_text,
        default_object=default_object,
        title=title,
    )
    app.run(jupyter_mode='external')
