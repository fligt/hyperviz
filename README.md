# Hyperviz

Hyperviz is ...

## Installation

From a checkout of this repository, create the development environment and
install Hyperviz with:

```bash
uv sync
```

## Choose a viewer

- [NPZ Viewer](notebooks/01_npz_viewer.qmd) provides a compact viewer for
  quickly inspecting hyperspectral cubes.
- [NPZ & TIF Dashboard](notebooks/02_npz_dashboard.qmd) displays NPZ and TIFF
  data together and supports drawing, importing, and exporting regions of
  interest.

![The Hyperviz NPZ and TIFF dashboard](images/npz_tif_dashboard.png)

## Basic usage

Load a data mapping and pass it to one of the public entry points:

```python
from fairdatanow import data_now
from hyperviz import make_dashboard

data = data_now(url, toml_text)
make_dashboard(data, toml_text)
```

The guide pages contain complete data-selection examples and automatically
generated API sections based on the current Python signatures and docstrings.
