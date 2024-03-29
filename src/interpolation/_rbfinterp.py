"""Module for RBF interpolation."""
import warnings
from itertools import combinations_with_replacement
from typing import Optional, List, Any, Tuple

import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError
from sklearn.neighbors import BallTree
from scipy.special import comb
from scipy.linalg.lapack import dgesv

from ._rbfinterp_pythran import (_build_system, _build_evaluation_coefficients, _polynomial_matrix)

float_like = npt.NDArray[np.floating[Any]]

__all__ = ["RBFInterpolator"]


# These RBFs are implemented.
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
    "haversine"
}


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic","cosine","haversine"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
    "cosine": 0,
    "haversine": 0
}


def _monomial_powers(ndim: int, degree: int):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos: int = comb(degree + ndim, ndim, exact=True) # type: ignore
    out = np.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    lhs, rhs, shift, scale = _build_system(
        y, d, smoothing, kernel, epsilon, powers
        )
    _, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
    if info < 0:
        raise ValueError(f"The {-info}-th argument had an illegal value.")
    elif info > 0:
        msg = "Singular matrix."
        nmonos = powers.shape[0]
        if nmonos > 0:
            pmat = _polynomial_matrix((y - shift)/scale, powers)
            rank = np.linalg.matrix_rank(pmat)
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                    )

        raise LinAlgError(msg)

    return shift, scale, coeffs


class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions.

    Parameters
    ----------
    y : (P, N) array_like
        Data point coordinates.
    d : (P, ...) array_like
        Data values at `y`.
    lon_column : int
        Column index of longitude values
    lat_column : int
        Column index of latitude values
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    min_points : int, optional
        Minimum amount of points within max_distance for the interpolation point
        to be valid,
    max_distance: float
        Maximum distance from grid point to interpolation point in km.
    earth_radius: float, optional
        Radius of earth in km. Default is 6371km
    smoothing : float or (P,) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

        Default is 'thin_plate_spline'.       
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2
    distance_to_time_scaling: float
        feature scale maximum time to maximum distance
    
        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of data values :math:`d`, which are from
    locations :math:`y`, is a linear combination of RBFs centered at :math:`y`
    plus a polynomial with a specified degree. The RBF interpolant is written
    as

    .. math::
        f(x) = K(x, y) a + P(x) b,

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    evaluated at the points :math:`x`, and :math:`P(x)` is a matrix of
    monomials, which span polynomials with the specified degree, evaluated at
    :math:`x`. The coefficients :math:`a` and :math:`b` are the solution to the
    linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a non-negative smoothing parameter that controls
    how well we want to fit the data. The data are fit exactly when the
    smoothing parameter is 0.

    The above system is uniquely solvable if the following requirements are
    met:

        - :math:`P(y)` must have full column rank. :math:`P(y)` always has full
          column rank when `degree` is -1 or 0. When `degree` is 1,
          :math:`P(y)` has full column rank if the data point locations are not
          all collinear (N=2), coplanar (N=3), etc.
        - If `kernel` is 'multiquadric', 'linear', 'thin_plate_spline',
          'cubic', or 'quintic', then `degree` must not be lower than the
          minimum value listed above.
        - If `smoothing` is 0, then each data point location must be distinct.

    When using an RBF that is not scale invariant ('multiquadric',
    'inverse_multiquadric', 'inverse_quadratic', or 'gaussian'), an appropriate
    shape parameter must be chosen (e.g., through cross validation). Smaller
    values for the shape parameter correspond to wider RBFs. The problem can
    become ill-conditioned or singular when the shape parameter is too small.

    The memory required to solve for the RBF interpolation coefficients
    increases quadratically with the number of data points, which can become
    impractical when interpolating more than about a thousand data points.
    To overcome memory limitations for large interpolation problems, the
    `neighbors` argument can be specified to compute an RBF interpolant for
    each evaluation point using only the nearest data points.

    .. versionadded:: 1.7.0

    See Also
    --------
    NearestNDInterpolator
    LinearNDInterpolator
    CloughTocher2DInterpolator

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    Examples
    --------
    Demonstrate interpolating scattered data to a grid in 2-D.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import RBFInterpolator
    >>> from scipy.stats.qmc import Halton

    >>> rng = np.random.default_rng()
    >>> xobs = 2*Halton(2, seed=rng).random(100) - 1
    >>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    >>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    >>> xflat = xgrid.reshape(2, -1).T
    >>> yflat = RBFInterpolator(xobs, yobs)(xflat)
    >>> ygrid = yflat.reshape(50, 50)

    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    >>> p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    >>> fig.colorbar(p)
    >>> plt.show()

    """

    def __init__(self,
        y: npt.ArrayLike,
        d: npt.ArrayLike,
        lon_column: int,
        lat_column: int,
        distance_to_time_scaling: float,
        max_distance: float,
        earth_radius: float = 6371.0,
        neighbors: Optional[int] = None,
        min_points: Optional[int] = None,
        smoothing: float | npt.ArrayLike = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: Optional[float] = None,
        degree: Optional[int] = None):
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

        d_dtype = complex if np.iscomplexobj(d) else float
        d = np.asarray(d, dtype=d_dtype, order="C")
        if d.shape[0] != ny:
            raise ValueError(
                f"Expected the first axis of `d` to have length {ny}."
            )

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        d = d.view(float)

        if np.isscalar(smoothing):
            smoothing = np.full(ny, smoothing, dtype=float)
        else:
            smoothing = np.asarray(smoothing, dtype=float, order="C")
            if smoothing.shape != (ny,):
                raise ValueError(
                    "Expected `smoothing` to be a scalar or have shape "
                    f"({ny},)."
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{_SCALE_INVARIANT}."
                )
        else:
            epsilon = float(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} when `kernel` "
                    f"is '{kernel}'. The interpolant may not be uniquely "
                    "solvable, and the smoothing parameter may have an "
                    "unintuitive effect.",
                    UserWarning
                )

        if neighbors is None:
            nobs = ny
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, ny))
            nobs = neighbors

        powers = _monomial_powers(ndim, degree)
        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}."
            )

        if neighbors is None:
            shift, scale, coeffs = _build_and_solve_system(
                y, d, smoothing, kernel, epsilon, powers
            )

            # Make these attributes private since they do not always exist.
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs

        else:
            self._tree = BallTree(y[:, [lat_column, lon_column]] * np.pi / 180, metric = 'haversine')

        self.y = y
        self.d = d
        self.min_points = min_points
        self.max_distance = max_distance
        self.earth_radius = earth_radius
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.powers = powers
        self.latlon_columns = [lat_column, lon_column]
        self.distance_to_time_scaling = distance_to_time_scaling

    def _chunk_evaluator(
            self,
            x: float_like,
            y: float_like,
            shift: float_like,
            scale: float_like,
            coeffs: float_like,
            memory_budget: int = 1000000
    ):
        """
        Evaluate the interpolation while controlling memory consumption.
        We chunk the input if we need more memory than specified.

        Parameters
        ----------
        x : (Q, N) float ndarray
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget: int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) float ndarray
        Interpolated array
        """
        nx, ndim = x.shape
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors
        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // ((self.powers.shape[0] + nnei)) + 1
        if chunksize <= nx:
            out = np.empty((nx, self.d.shape[1]), dtype=float)
            for i in range(0, nx, chunksize):
                vec = _build_evaluation_coefficients(
                    x[i:i + chunksize, :],
                    y,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale)
                out[i:i + chunksize, :] = np.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(
                x,
                y,
                self.kernel,
                self.epsilon,
                self.powers,
                shift,
                scale)
            out = np.dot(vec, coeffs)
        return out

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
            raise ValueError("Expected the second axis of `x` to have length "
                             f"{self.y.shape[1]}.")

        # Our memory budget for storing RBF coefficients is
        # based on how many floats in memory we already occupy
        # If this number is below 1e6 we just use 1e6
        # This memory budget is used to decide how we chunk
        # the inputs
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

        if self.neighbors is None:
            out = self._chunk_evaluator(
                x,
                self.y,
                self._shift,
                self._scale,
                self._coeffs,
                memory_budget=memory_budget)
        else:
            # Setup coordinate
            x, y_new, yindices, valid_yindices = self.setup_coordinates(x)

            # Setup output
            out = np.zeros((nx, self.d.shape[1]), dtype=float)
            out[~valid_yindices] = np.nan

            # Interpolate data
            sub_out = np.zeros((x.shape[0], self.d.shape[1]), dtype=float)

            for xidx, yidx in enumerate(yindices):
                # `yidx` are the indices of the observations in this
                # neighborhood. `xidx` are the indices of the evaluation points
                # that are using this neighborhood.
                xnbr = x[xidx:xidx+1]
                ynbr = y_new[xidx]
                dnbr = self.d[yidx]
                snbr = self.smoothing[yidx]
                shift, scale, coeffs = _build_and_solve_system(
                    ynbr,
                    dnbr,
                    snbr,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                )
                sub_out[xidx] = self._chunk_evaluator(
                    xnbr,
                    ynbr,
                    shift,
                    scale,
                    coeffs,
                    memory_budget=memory_budget)
            out[valid_yindices] = sub_out
        out = out.view(self.d_dtype)
        out = out.reshape((nx, ) + self.d_shape)
        return out

    def feature_scale(self, x: float_like, yindices: npt.NDArray[Any], distance_km: npt.NDArray[Any]) -> List[float_like]:
        # Feature scale around 0
        x[:, -1] = 0

        # Maximum distance in distance_km
        max_dist = max([dist.max() for dist in distance_km])

        # Get min max feature values (a, b) (Distances)
        min_dist = 0
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
                        * (max_dist_scaled - min_dist)
                    )
                    / y_diff
                )

        # Add a to feature scaled
        return [y + min_dist for y in y_valids]

    @staticmethod
    def random_sample(distances_scaled: npt.NDArray[Any], n_samples: int) -> List[npt.NDArray[np.int64]]:
        """Randomly samples n_samples from distances_scaled if atleast n_samples are present.
        Otherwise the entire vector is keept.
        The output the the sampled indexes of each numpy array in distances_scaled.
        distances_scaled have to be between 0 and 1 for all values.
        """
        sampled_values = []
        for distance_scaled in distances_scaled:
            if (n_points := len(distance_scaled)) > n_samples:
                indexes = np.arange(n_points)
                total_dist = (1 - distance_scaled).sum()
                probability = (1 - distance_scaled)/total_dist
                samples = np.random.choice(indexes, n_samples, replace=False, p=probability)
                sampled_values.append(np.in1d(indexes, samples, assume_unique=True))
            else:
                sampled_values.append(np.ones(n_points, dtype=bool))
        return sampled_values


    def setup_coordinates(self, x: float_like) -> Tuple[
        float_like, List[float_like], npt.NDArray[Any], npt.NDArray[np.bool_]
    ]:
        """Find valid x_coordinates and y_indices"""
        # Get the indices of the k nearest observation points to each
        # evaluation point.
        x_latlon = x[:, self.latlon_columns]
        yindices: npt.NDArray[Any]
        distances_scaled: npt.NDArray[Any]
        yindices, distances_scaled = self._tree.query_radius(
            x_latlon * np.pi / 180,
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

        # Sample from tree
        # if self.neighbors is not None:
        #     sampled = self.random_sample(distances_scaled, self.neighbors)
        #     new_yindices = []
        #     new_distances_scaled = []
        #     for sample, yindice, distance_scaled in zip(sampled, yindices, distances_scaled):
        #         new_yindices.append(yindice[sample])
        #         new_distances_scaled.append(distance_scaled[sample])
        #     yindices = np.array(new_yindices, dtype='object')
        #     distances_scaled = np.array(new_distances_scaled, dtype='object')

        # Based on distance
        if self.neighbors is not None:
            yindices_list = [y_idx[:self.neighbors] for y_idx in yindices]
            first_len = len(yindices_list[0])
            if all((len(yindice) == first_len for yindice in yindices_list)):
                dtype = np.int64
                d_dtype = np.float64
            else:
                dtype = 'object'
                d_dtype = 'object'
            yindices = np.array(yindices_list, dtype=dtype)

            distances_scaled = np.array([dist[:self.neighbors] for dist in distances_scaled], dtype=d_dtype)

        # Convert from 0-1 distances to km
        distances_km = distances_scaled * self.earth_radius

        # Feature scale
        y_new = self.feature_scale(x, yindices, distances_km)

        # Feature scale
        return x, y_new, yindices, valid_grid_points