import numpy as np

# Kernels
def linear(r):
    return -r

def thin_plate_spline(r):
    if r == 0:
        return 0.0
    else:
        return r**2*np.log(r)

def cubic(r):
    return r**3

def quintic(r):
    return -r**5

def multiquadric(r):
    return -np.sqrt(r**2 + 1)

def inverse_multiquadric(r):
    return 1/np.sqrt(r**2 + 1)

def inverse_quadratic(r):
    return 1/(r**2 + 1)

def gaussian(r):
    return np.exp(-r**2)

def euclidean(x, y):
    return np.linalg.norm(x - y)

def cosine(x, y):
    dot_p = np.dot(x, y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    return 1 - (dot_p / (x_norm * y_norm)) 

# Distance measures
DISTANCE_NAME_TO_FUNC = {
    "cosine" : cosine,
    "euclidean" : euclidean,
}

KERNEL_NAME_TO_FUNC = {
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian
}

def kernel_vector(x, y, kernel_func, dist_func, out):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in range(y.shape[0]):
        out[i] = kernel_func(dist_func(x, y[i]))

def polynomial_vector(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in range(powers.shape[0]):
        out[i] = np.prod(x**powers[i])

def kernel_matrix(x, kernel_func, dist_func, out):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(i+1):
            out[i, j] = kernel_func(dist_func(x[i], x[j]))
            out[j, i] = out[i, j]

def polynomial_matrix(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i]**powers[j])

# pythran export _kernel_matrix(float[:, :], str, str)
def _kernel_matrix(x, kernel, distance):
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    out = np.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = KERNEL_NAME_TO_FUNC[kernel]
    dist_func = DISTANCE_NAME_TO_FUNC[distance]
    kernel_matrix(x, kernel_func, dist_func, out)
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
#                              str,
#                              float,
#                              int[:, :])
def _build_system(y, d, smoothing, kernel, distance, epsilon, powers):
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
    distane : str
        Name of the distance formula.
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
    kernel_func = KERNEL_NAME_TO_FUNC[kernel]
    dist_func = DISTANCE_NAME_TO_FUNC[distance]

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
    kernel_matrix(yeps, kernel_func, dist_func, lhs[:p, :p])
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
#                          str,
#                          float,
#                          int[:, :],
#                          float[:],
#                          float[:])
def _build_evaluation_coefficients(x, y, kernel, distance, epsilon, powers, shift, scale):
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
    distane : str
        Name of the distance formula.
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
    kernel_func = KERNEL_NAME_TO_FUNC[kernel]
    dist_func = DISTANCE_NAME_TO_FUNC[distance]

    yeps = y*epsilon
    xeps = x*epsilon
    xhat = (x - shift)/scale


    x = np.array([[ 3.124626, 37.543167],
        [ 3.18617,  37.636817],
        [ 3.217007, 37.683626],])
    a = np.zeros((1,2))
    powers =  np.array([[2,2]])
    polynomial_vector(x, powers, a)

    vec = np.empty((q, p + r), dtype=float)
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, dist_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])

    return vec


# pythran export _full_evaluator(float[:, :],
#                          float[:, :],
#                          float[:],
#                          float[:],
#                          float[:, :],
#                          int[:, :],
#                          str,
#                          str,
#                          float)
def _full_evaluator(
    x,
    y,
    shift,
    scale,
    coeffs,
    powers,
    kernel,
    distance,
    epsilon,
):  
    vec = _build_evaluation_coefficients(
        x,
        y,
        kernel,
        distance,
        epsilon,
        powers,
        shift,
        scale
    )
    return np.dot(vec, coeffs)

# pythran export _chunk_evaluator(float[:, :],
#                          float[:, :],
#                          float[:],
#                          float[:],
#                          float[:, :],
#                          int,
#                          int[:, :],
#                          str,
#                          str,
#                          float,
#                          float[:, :])
def _chunk_evaluator(
    x,
    y,
    shift,
    scale,
    coeffs,
    chunksize,
    powers,
    kernel,
    distance,
    epsilon,
    d
):
    nx, _ = x.shape
    # in each chunk we consume the same space we already occupy
    out = np.empty((nx, d.shape[1]), dtype=float)
    for i in range(0, nx, chunksize):
        out[i:i + chunksize, :] = _full_evaluator(x[i:i + chunksize, :],
            y,
            shift,
            scale,
            coeffs,
            powers,
            kernel,
            distance,
            epsilon,
        )
    return out


# pythran export _build_and_solve_system(float[:, :],
#                              float[:, :],
#                              float[:],
#                              str,
#                              str,
#                              float,
#                              int[:, :])
def _build_and_solve_system(
    y,
    d,
    smoothing,
    kernel,
    distance,
    epsilon,
    powers
):
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
    
    input_matrix = lhs.T @ lhs
    output_matrix = np.zeros((len(input_matrix), len(input_matrix)), dtype=np.float64)
    #inv(input_matrix, output_matrix)
    coeffs = np.linalg.inv(input_matrix) @ lhs.T @ rhs
    return shift, scale, coeffs

# pythran export _interpolate_neighbors(float[:, :],
#                              float[:, :],
#                              float[:, :],
#                              float[:, :],
#                              float[:],
#                              int,
#                              str,
#                              str,
#                              float,
#                              int[:, :],
#                              int,
#                              int)
def _interpolate_neighbors(yindices, d, x, y, smoothing, nx, kernel, distance, epsilon, powers, memory_budget, neighbors):
    # Multiple evaluation points may have the same neighborhood of
    # observation points. Make the neighborhoods unique so that we only
    # compute the interpolation coefficients once for each
    # neighborhood.
    yindices = np.sort(yindices, axis=1)
    yindices, inv_ = np.unique(yindices, return_inverse=True, axis=0)
    # `inv` tells us which neighborhood will be used by each evaluation
    # point. Now we find which evaluation points will be using each
    # neighborhood.
    xindices = [[] for _ in range(len(yindices))]
    for i, j in enumerate(inv_):
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


def getCfactor(M, t, p, q, n):
    i = 0
    j = 0
    for r in range(n):
        for c in range(n):
            if (r != p and c != q):
                t[i][j] = M[r][c]
                j += 1
                if (j == n - 1):
                    j = 0
                    i += 1

def determinant(M, n):
    D = 0
    if (n == 1):
      return M[0][0]
    t = np.zeros((n,n), dtype=np.float64)
    s = 1
    for f in range(n):
        getCfactor(M, t, 0, f, n)
        D += s * M[0][f] * determinant(t, n - 1)
        s = -s
    return D

def adjugate(M, adj):
    N = len(M)
    if (N == 1):
        adj[0][0] = 1
        return

    s = 1,
    t = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            getCfactor(M, t, i, j, N)
            s = 1 if ((i+j)%2==0) else -1
            adj[j][i] = (s)*(determinant(t, N-1))

def inv(M, inv):
    N = len(M)
    det = determinant(M, N)
    if (det == 0):
        return False

    adj = np.zeros((N, N), dtype=np.float64)
    adjugate(M, adj)

    inv[:,:] = adj/det
    return True