"""Module for RBF interpolation."""
import warnings
from itertools import combinations_with_replacement
from typing import Optional, Tuple

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError
from scipy.spatial import KDTree
from scipy.special import comb
from scipy.linalg.lapack import dgesv  # type: ignore[attr-defined]
from ._rbfinterp import (_build_system, _full_evaluator, _polynomial_matrix)
from ._rbfinterp_wrap import _interpolate_neighbors

# These RBFs are implemented.
_KERNEL_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
}
# These distances are implemented.
_DISTANCE_AVAILABLE = {
    "cosine",
    "euclidean",
}
# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}

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
    "quintic": 2
}

def _monomial_powers(ndim, degree):
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

def _build_and_solve_system(
    y: npt.NDArray,
    d: npt.NDArray,
    smoothing: npt.NDArray[np.float64],
    kernel: str,
    distance: str,
    epsilon: float,
    powers: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    distane : str
        Name of the distance formula.
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
        y, d, smoothing, kernel, distance, epsilon, powers
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
    def __init__(
        self,
        y: npt.NDArray,
        d: npt.NDArray,
        neighbors: Optional[int] = None,
        smoothing: float | npt.NDArray = 0.0,
        kernel: str = "thin_plate_spline",
        distance: str = "euclidean",
        epsilon: Optional[float] = None,
        degree: Optional[float] = None
    ):
        y = np.asarray(y, dtype=float, order="C")
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")

        ny, ndim = y.shape

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
        if kernel not in _KERNEL_AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_KERNEL_AVAILABLE}.")
        distance = distance.lower()
        if distance not in _DISTANCE_AVAILABLE:
            raise ValueError(f"`distance` must be one of {_DISTANCE_AVAILABLE}.")

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
                y, d, smoothing, kernel, distance, epsilon, powers
            )

            # Make these attributes private since they do not always exist.
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs

        else:
            self._tree = KDTree(y)

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.distance = distance
        self.epsilon = epsilon
        self.powers = powers

    # def _chunk_evaluator(
    #         self,
    #         x,
    #         y,
    #         shift,
    #         scale,
    #         coeffs,
    #         memory_budget=1000000
    #     ):
    #     """
    #     Evaluate the interpolation while controlling memory consumption.
    #     We chunk the input if we need more memory than specified.

    #     Parameters
    #     ----------
    #     x : (Q, N) float ndarray
    #         array of points on which to evaluate
    #     y: (P, N) float ndarray
    #         array of points on which we know function values
    #     shift: (N, ) ndarray
    #         Domain shift used to create the polynomial matrix.
    #     scale : (N,) float ndarray
    #         Domain scaling used to create the polynomial matrix.
    #     coeffs: (P+R, S) float ndarray
    #         Coefficients in front of basis functions
    #     memory_budget: int
    #         Total amount of memory (in units of sizeof(float)) we wish
    #         to devote for storing the array of coefficients for
    #         interpolated points. If we need more memory than that, we
    #         chunk the input.

    #     Returns
    #     -------
    #     (Q, S) float ndarray
    #     Interpolated array
    #     """
        
    #     nx, _ = x.shape
    #     if self.neighbors is None:
    #         nnei = len(y)
    #     else:
    #         nnei = self.neighbors
    #     # in each chunk we consume the same space we already occupy
    #     chunksize = memory_budget // ((self.powers.shape[0] + nnei)) + 1
    #     if chunksize <= nx:
    #         out = np.empty((nx, self.d.shape[1]), dtype=float)
    #         for i in range(0, nx, chunksize):
    #             vec = _build_evaluation_coefficients(
    #                 x[i:i + chunksize, :],
    #                 y,
    #                 self.kernel,
    #                 self.distance,
    #                 self.epsilon,
    #                 self.powers,
    #                 shift,
    #                 scale)
    #             out[i:i + chunksize, :] = np.dot(vec, coeffs)
    #     else:
    #         vec = _build_evaluation_coefficients(
    #             x,
    #             y,
    #             self.kernel,
    #             self.distance,
    #             self.epsilon,
    #             self.powers,
    #             shift,
    #             scale
    #         )
    #         out = np.dot(vec, coeffs)
    #     return out

    def __call__(self, x):
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
            out = _full_evaluator(
                x,
                self.y,
                self._shift,
                self._scale,
                self._coeffs,
                self.powers,
                self.kernel,
                self.distance,
                self.epsilon
            )
        else:
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            _, yindices = self._tree.query(x, self.neighbors)
            if self.neighbors == 1:
                # `KDTree` squeezes the output when neighbors=1.
                yindices = yindices[:, None] # type: ignore

            # Multiple evaluation points may have the same neighborhood of
            # observation points. Make the neighborhoods unique so that we only
            # compute the interpolation coefficients once for each
            # neighborhood.
            # yindices = np.sort(yindices, axis=1)
            # yindices, inv = np.unique(yindices, return_inverse=True, axis=0)
            # # `inv` tells us which neighborhood will be used by each evaluation
            # # point. Now we find which evaluation points will be using each
            # # neighborhood.
            # xindices = [[] for _ in range(len(yindices))]
            # for i, j in enumerate(inv):
            #     xindices[j].append(i)

            # out = np.empty((nx, self.d.shape[1]), dtype=float)
            # for xidx, yidx in tqdm(zip(xindices, yindices), total=min([len(xindices), len(yindices)])):
            #     # `yidx` are the indices of the observations in this
            #     # neighborhood. `xidx` are the indices of the evaluation points
            #     # that are using this neighborhood.
            #     xnbr = x[xidx]
            #     ynbr = self.y[yidx]
            #     dnbr = self.d[yidx]
            #     snbr = self.smoothing[yidx]
            #     shift, scale, coeffs = _build_and_solve_system(
            #         ynbr,
            #         dnbr,
            #         snbr,
            #         self.kernel,
            #         self.distance,
            #         self.epsilon,
            #         self.powers,
            #     )
            #     out[xidx] = self._chunk_evaluator(
            #         xnbr,
            #         ynbr,
            #         shift,
            #         scale,
            #         coeffs,
            #         memory_budget=memory_budget
            #     )
            out = _interpolate_neighbors(yindices, self.d, x, self.y, self.smoothing, nx, self.kernel, self.distance, self.epsilon, self.powers, memory_budget, self.neighbors)
        out = out.view(self.d_dtype)
        out = out.reshape((nx, ) + self.d_shape)
        return out
