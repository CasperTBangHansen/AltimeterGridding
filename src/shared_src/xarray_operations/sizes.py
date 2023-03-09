from typing import Tuple, Iterable, Hashable
import xarray as xr

def transform(dataset: xr.Dataset) -> Tuple[float, float, float, float, float, float]:
    """
    Finds the transform (affine) of the `xarray.Dataset` | `xarray.DataArray`
    """
    try:
        src_left, _, _, src_top = unordered_bounds(dataset)
        src_resolution_x, src_resolution_y = resolution(dataset)
    except (ValueError, IndexError):
        return (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) # Identity affine
    return (src_left, src_resolution_x, 0.0, src_top, 0.0, src_resolution_y)

def unordered_bounds(dataset: xr.Dataset) -> Tuple[float, float, float, float]:
    """
    Unordered bounds of Outermost coordinates of the `xarray.DataArray` | `xarray.Dataset`
    The return form is (left, bottom, right, top)
    """
    resolution_x, resolution_y = resolution(dataset)
    left, bottom, right, top = internal_bounds(dataset)
    left -= resolution_x / 2.0
    right += resolution_x / 2.0
    top -= resolution_y / 2.0
    bottom += resolution_y / 2.0
    return (left, bottom, right, top)

def internal_bounds(dataset: xr.Dataset) -> Tuple[float, float, float, float]:
    """
    Determine the internal bounds of the `xarray.DataArray`.
    The return format is (left, bottom, right, top)
    """
    x_dim, y_dim = get_dimension_names(dataset)
    try:
        # Get boundaries
        boundary: Iterable[float] = []
        for index in [0, -1]:
            for dim in [x_dim, y_dim]:
                boundary.append(dataset[dim][index].item())
    except IndexError as e:
        raise IndexError("Unable to determine bounds from coordinates.") from e
    return tuple(boundary)

def resolution(dataset: xr.Dataset) -> Tuple[float, float]:
    """
    Determine if the resolution of the grid.
    If the transformation has rotation, the sign of the resolution is lost.
    Returns (x_resolution, y_resolution)
    """
    # Boundary
    left, bottom, right, top = internal_bounds(dataset)
    
    # Width/Height
    x_dim, y_dim = get_dimension_names(dataset)
    width = dataset[x_dim].size
    height = dataset[y_dim].size

    # Resolutions
    resolution_x = (right - left) / (width - 1)
    resolution_y = (bottom - top) / (height - 1)
    return resolution_x, resolution_y

def get_dimension_names(dataset: xr.Dataset) -> Tuple[Hashable, Hashable]:
    """
    Finds the x and y dimensions of the dataset.
    Returns (x_dim, y_dim)
    """
    x_dim, y_dim = list(dataset.sizes)
    if x_dim not in dataset.coords:
        raise ValueError(f"{x_dim} missing coordinates.")
    if y_dim not in dataset.coords:
        raise ValueError(f"{y_dim} missing coordinates.")
    return (x_dim, y_dim)
