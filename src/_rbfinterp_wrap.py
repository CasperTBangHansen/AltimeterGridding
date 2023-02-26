import numpy as np
from ._rbfinterp import _build_system, _chunk_evaluator, _full_evaluator, _build_and_solve_system

# def _build_and_solve_system(
#     y,
#     d,
#     smoothing,
#     kernel,
#     distance,
#     epsilon,
#     powers
# ):
#     """Build and solve the RBF interpolation system of equations.

#     Parameters
#     ----------
#     y : (P, N) float ndarray
#         Data point coordinates.
#     d : (P, S) float ndarray
#         Data values at `y`.
#     smoothing : (P,) float ndarray
#         Smoothing parameter for each data point.
#     kernel : str
#         Name of the RBF.
#     distane : str
#         Name of the distance formula.
#     epsilon : float
#         Shape parameter.
#     powers : (R, N) int ndarray
#         The exponents for each monomial in the polynomial.
#     Returns
#     -------
#     coeffs : (P + R, S) float ndarray
#         Coefficients for each RBF and monomial.
#     shift : (N,) float ndarray
#         Domain shift used to create the polynomial matrix.
#     scale : (N,) float ndarray
#         Domain scaling used to create the polynomial matrix.

#     """
#     lhs, rhs, shift, scale = _build_system(
#         y, d, smoothing, kernel, distance, epsilon, powers
#     )
#     #coeffs = solve(lhs, rhs)
#     coeffs = np.linalg.inv(lhs.T @ lhs) @ lhs.T @ rhs
#     # (lhs.T @ lhs)**(-1) @ lhs.T @ rhs
#     # (G.T @ G)**(-1)G.T @ X
#     return shift, scale, coeffs

def _interpolate_neighbors(yindices, d, x, y, smoothing, nx, kernel, distance, epsilon, powers, memory_budget, neighbors):
    # Multiple evaluation points may have the same neighborhood of
    # observation points. Make the neighborhoods unique so that we only
    # compute the interpolation coefficients once for each
    # neighborhood.
    yindices = np.sort(yindices, axis=1)
    yindices, inv = np.unique(yindices, return_inverse=True, axis=0)
    # `inv` tells us which neighborhood will be used by each evaluation
    # point. Now we find which evaluation points will be using each
    # neighborhood.
    xindices = [[] for _ in range(len(yindices))]
    for i, j in enumerate(inv):
        xindices[j].append(i)

    chunksize = memory_budget // ((powers.shape[0] + neighbors)) + 1
    out = np.empty((nx, d.shape[1]), dtype=float)
    for xidx, yidx in zip(xindices, yindices):
        # `yidx` are the indices of the observations in this
        # neighborhood. `xidx` are the indices of the evaluation points
        # that are using this neighborhood.
        xnbr = x[xidx]
        ynbr = y[yidx]
        dnbr = d[yidx]
        snbr = smoothing[yidx]
        shift, scale, coeffs = _build_and_solve_system(
            ynbr,
            dnbr,
            snbr,
            kernel,
            distance,
            epsilon,
            powers,
        )
        if chunksize <= nx:
            out[xidx] = _chunk_evaluator(
                xnbr,
                ynbr,
                shift,
                scale,
                coeffs,
                chunksize,
                powers,
                kernel,
                distance,
                epsilon,
                d
            )
        else:
            out[xidx] = _full_evaluator(
                xnbr,
                ynbr,
                shift,
                scale,
                coeffs,
                powers,
                kernel,
                distance,
                epsilon
            )
    return out