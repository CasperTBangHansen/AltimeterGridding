import numpy as np


def linear(x, y):
    return -np.linalg.norm(x - y)


def thin_plate_spline(x, y):
    r = np.linalg.norm(x - y)
    if r == 0:
        return 0.0
    else:
        return r**2*np.log(r)


def cubic(x, y):
    return np.linalg.norm(x - y)**3

def quintic(x, y):
    return -np.linalg.norm(x - y)**5


def multiquadric(x, y):
    return -np.sqrt(np.linalg.norm(x - y)**2 + 1)


def inverse_multiquadric(x, y):
    return 1/np.sqrt(np.linalg.norm(x - y)**2 + 1)


def inverse_quadratic(x, y):
    return 1/(np.linalg.norm(x - y)**2 + 1)


def gaussian(x, y):
    return np.exp(-np.linalg.norm(x - y)**2)


def cosine(x, y):
    return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


NAME_TO_FUNC = {
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian,
   "cosine": cosine
}


def kernel_vector(x, y, kernel_func, out):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = kernel_func(x, y[i])


def polynomial_vector(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in range(powers.shape[0]):
        out[i] = np.prod(x**powers[i])


def kernel_matrix(x, kernel_func, out):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(i+1):
            out[i, j] = kernel_func(x[i], x[j])
            out[j, i] = out[i, j]


def polynomial_matrix(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i]**powers[j])


# pythran export _kernel_matrix(float[:, :], str)
def _kernel_matrix(x, kernel):
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    out = np.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = NAME_TO_FUNC[kernel]
    kernel_matrix(x, kernel_func, out)
    return out


# pythran export _polynomial_matrix(float[:, :], int[:, :])
def _polynomial_matrix(x, powers):
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    out = np.empty((x.shape[0], powers.shape[0]), dtype=float)
    polynomial_matrix(x, powers, out)
    return out


# pythran export _build_system(float[:, :],
#                              float[:, :],
#                              float[:],
#                              str,
#                              float,
#                              int[:, :])
def _build_system(y, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients.

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
    lhs : (P + R, P + R) float ndarray
        Left-hand side matrix.
    rhs : (P + R, S) float ndarray
        Right-hand side matrix.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    # Shift and scale the polynomial domain to be between -1 and 1
    mins = np.min(y, axis=0)
    maxs = np.max(y, axis=0)
    shift = (maxs + mins)/2
    scale = (maxs - mins)/2
    # The scale may be zero if there is a single point or all the points have
    # the same value for some dimension. Avoid division by zero by replacing
    # zeros with ones.
    scale[scale == 0.0] = 1.0

    yeps = y*epsilon
    yhat = (y - shift)/scale

    # Transpose to make the array fortran contiguous. This is required for
    # dgesv to not make a copy of lhs.
    lhs = np.empty((p + r, p + r), dtype=float).T
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    for i in range(p):
        lhs[i, i] += smoothing[i]

    # Transpose to make the array fortran contiguous.
    rhs = np.empty((s, p + r), dtype=float).T
    rhs[:p] = d
    rhs[p:] = 0.0

    return lhs, rhs, shift, scale


# pythran export _build_evaluation_coefficients(float[:, :],
#                          float[:, :],
#                          str,
#                          float,
#                          int[:, :],
#                          float[:],
#                          float[:])
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers,
                                   shift, scale):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) float ndarray

    """
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    yeps = y*epsilon
    xeps = x*epsilon
    xhat = (x - shift)/scale

    vec = np.empty((q, p + r), dtype=float)
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])

    return vec
