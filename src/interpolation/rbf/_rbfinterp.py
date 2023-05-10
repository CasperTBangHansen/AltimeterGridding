"""Module for RBF interpolation."""
from typing import Optional, List, Any, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import BallTree
from scipy.linalg.lapack import dgesv

from time import perf_counter

float_like = npt.NDArray[np.floating[Any]]

__all__ = ["RBFInterpolator"]


def build_and_evaluate(y: float_like, t: float_like, x: float_like) -> float:
    """Build and solve the RBF interpolation system of equations"""
    lat1 = y[:,1][:, np.newaxis]
    lon1 = y[:,0][:, np.newaxis]
    lat2 = y[:,1]
    lon2 = y[:,0]
    dtime = y[:, 2] - y[:, 2][:, np.newaxis]

    # Compute kernel (A)
    a_matrix = np.sqrt(
        (0.1
         * (
             2*6371*np.arcsin(
                np.sqrt(
                    np.sin((lat2 - lat1)/2)**2
                    + np.cos(lat1)
                    * np.cos(lat2)
                    * np.sin((lon2 - lon1)/2)**2
                )
            )
         )
        )**2
        + (0.5625 * dtime)**2
    )
    
    # Find weights
    _, _, weights, _ = dgesv(a_matrix, t, overwrite_a=False, overwrite_b=False)
    
    # Compute A matrix to interpolate to
    a_matrix = np.sqrt(
        (0.1
         * (
             2*6371*np.arcsin(
                np.sqrt(
                    np.sin((lat2 - x[1])/2)**2
                    + np.cos(x[1])
                    * np.cos(lat2)
                    * np.sin((lon2 - x[0])/2)**2
                )
            )
         )
        )**2
        + (0.5625 * (y[:, 2] - x[2]))**2
    )
    
    # Interpolate
    return np.dot(a_matrix, weights)


class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions"""

    def __init__(
        self,
        y: npt.ArrayLike,
        t: npt.ArrayLike,
        lon_column: int,
        lat_column: int,
        distance_to_time_scaling: float,
        max_distance: float,
        earth_radius: float = 6371.0,
        neighbors: Optional[int] = None,
        min_points: Optional[int] = None
    ):
        y = np.asarray(y, dtype=float, order="C")
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")

        ny, ndim = y.shape

        if ndim < lat_column + 1:
            raise ValueError(
                f"Invalid `lat_column` was {lat_column} should have been less than {ndim}."
            )
        if ndim < lon_column + 1:
            ValueError(
                f"Invalid `lon_column` was {lon_column} should have been less than {ndim}."
            )

        t_dtype = complex if np.iscomplexobj(t) else float
        t = np.asarray(t, dtype=t_dtype, order="C")
        if t.shape[0] != ny:
            raise ValueError(
                f"Expected the first axis of `d` to have length {ny}."
            )

        t_shape = t.shape[1:]
        t = t.reshape((ny, -1))

        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        t = t.view(float)

        # Convert to radians
        y[:, [lat_column, lon_column]] *= np.pi / 180

        if neighbors is None:
            neighbors = ny
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, ny))
        
        # Fit tree
        self._tree = BallTree(y[:, [lat_column, lon_column]], metric = 'haversine')

        self.y = y
        self.t = t
        self.min_points = min_points
        self.max_distance = max_distance
        self.earth_radius = earth_radius
        self.t_shape = t_shape
        self.t_dtype = t_dtype
        self.neighbors = neighbors
        self.latlon_columns = [lat_column, lon_column]
        self.distance_to_time_scaling = distance_to_time_scaling

    def __call__(self, x: float_like) -> float_like:
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """
        x = np.asarray(x, dtype=float, order="C")
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(f"Expected the second axis of `x` to have length {self.y.shape[1]}.")

        # Setup coordinate
        x, y_new, yindices, valid_yindices = self.setup_coordinates(x)

        # Setup output
        out = np.zeros((nx, self.t.shape[1]), dtype=float)
        out[~valid_yindices] = np.nan

        # Interpolate data
        sub_out = np.zeros((x.shape[0], self.t.shape[1]), dtype=float)
        
        t1_start = perf_counter()
        for xidx, yidx in enumerate(yindices):
            sub_out[xidx] = build_and_evaluate(
                x = x[xidx],
                y = y_new[xidx],
                t = self.t[yidx]
            )
        t1_end = perf_counter()
        print(f"\n\nTook {(t1_end - t1_start)/60} minutes and had {len(yindices)} iterations")

        out[valid_yindices] = sub_out
        out = out.view(self.t_dtype)
        out = out.reshape((nx, ) + self.t_shape)
        return out

    def feature_scale(self, x: float_like, yindices: npt.NDArray[Any], distance_km: npt.NDArray[Any]) -> List[float_like]:
        # Feature scale around 0
        x[:, -1] = 0

        # Maximum distance in distance_km
        max_dist = max([dist.max() for dist in distance_km])

        # Get min max feature values (a, b) (Distances)
        max_dist_scaled = self.distance_to_time_scaling * max_dist

        # Map y indices to y values
        y_valids: List[npt.NDArray[np.floating[Any]]] = [self.y[y_idx] for y_idx in yindices]
        
        for i, y_valid in enumerate(y_valids):
            # Compute min max time values
            y_min = y_valid[:, -1].min()
            y_diff = y_valid[:, -1].max() - y_min

            # Prevent division by zero
            if y_diff != 0:
                # Compute minmax feature scaling
                y_valids[i][:, -1] = (
                    (
                        (y_valid[:, -1] - y_min)
                        * max_dist_scaled
                    )
                    / y_diff
                )

        return y_valids

    def setup_coordinates(self, x: float_like) -> Tuple[
        float_like, List[float_like], npt.NDArray[Any], npt.NDArray[np.bool_]
    ]:
        """Find valid x_coordinates and y_indices"""
        # Get the indices of the k nearest observation points to each
        # evaluation point.
        x[:, self.latlon_columns] *= np.pi / 180
        yindices: npt.NDArray[Any]
        distances_scaled: npt.NDArray[Any]
        yindices, distances_scaled = self._tree.query_radius(
            x[:, self.latlon_columns],
            r = self.max_distance/self.earth_radius,
            return_distance = True,
            sort_results = True,
        )
        
        if self.neighbors == 1:
            # `KDTree` squeezes the output when neighbors=1.
            yindices = yindices[:, None]
            distances_scaled = distances_scaled[:, None]          
        
        min_points = 0 if self.min_points is None else self.min_points

        # Remove grid points for x and y if x is too far away from y.
        valid_grid_points = np.array([len(y) for y in yindices]) > min_points
        x = x[valid_grid_points]
        yindices = yindices[valid_grid_points]
        distances_scaled = distances_scaled[valid_grid_points]

        # Based on distance
        if self.neighbors is not None:
            yindices = np.array([y_idx[:self.neighbors] for y_idx in yindices], dtype='object')
            distances_scaled = np.array([dist[:self.neighbors ] for dist in distances_scaled], dtype='object')

        # Convert from 0-1 distances to km
        distances_km = distances_scaled * self.earth_radius

        # Feature scale
        y_new = self.feature_scale(x, yindices, distances_km)

        # Feature scale
        return x, y_new, yindices, valid_grid_points