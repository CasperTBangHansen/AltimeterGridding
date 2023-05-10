import numpy as np

# pythran export haversine_distance_matrix(float[:], float[:], float[:], float[:])
def haversine_distance_matrix(lat1, lat2, lon1, lon2):
    return 2*6371*np.arcsin(
        np.sqrt(
            np.square(
                np.sin((lat2.T-lat1)/2)
            )
            + np.cos(lat1)*np.cos(lat2.T)
            * np.square(
                np.sin((lon2.T-lon1)/2)
            )
        )
    )

# pythran export haversine_distance_vector(float[:], float, float[:], float)
def haversine_distance_vector(lat1, lat2, lon1, lon2):
    return 2*6371*np.arcsin(
        np.sqrt(
            np.square(
                np.sin((lat2-lat1)/2)
            )
            + np.cos(lat1)*np.cos(lat2)
            * np.square(
                np.sin((lon2-lon1)/2)
            )
        )
    )

# pythran export haversine_distance(float, float, float, float)
def haversine_distance(lat1, lat2, lon1, lon2):
    return 2*6371*np.arcsin(
        np.sqrt(
            np.square(
                np.sin((lat2-lat1)/2)
            )
            + np.cos(lat1)*np.cos(lat2)
            * np.square(
                np.sin((lon2-lon1)/2)
            )
        )
    )


# pythran export haversine_kernel(float[:, :])
def haversine_kernel(y):
    out = np.zeros((y.shape[0], y.shape[0]), dtype=float)
    for i in range(y.shape[0]):
        for j in range(i):
            out[i, j] = haversine(y[i], y[j])
            out[j, i] = out[i, j]
    return out

# pythran export haversine_kernel(float[:, :])
def haversine_kernel(y):
    out = np.zeros((y.shape[0], y.shape[0]), dtype=float)
    for i in range(y.shape[0] - 1):
        out[i+1:, i] = haversine_vector(y, y[i])
        out[i, i+1:] = out[i+1:, i]
    return out

# pythran export haversine_matrix(float[:, :], float[:, :])
def haversine_matrix(x, y):
    out = np.empty((x.shape[0], y.shape[0]), dtype=float)
    for i in range(y.shape[0]):
        out[:, i] = haversine_vector(x, y[i])
        out[i, i+1:] = out[i+1:, i]
    return out

# pythran export haversine_vector(float[:, :], float[:])
def haversine_vector(x, y):
    lon1, lat1 = x[:, 0] * np.pi/180, x[:, 1] * np.pi/180
    lon2, lat2 = y[0] * np.pi/180, y[1] * np.pi/180
    delta_t = x[:, 2] - y[2]
    return -np.sqrt(
        np.square(.1 * haversine_distance_vector(lat1, lat2, lon1, lon2))
        + np.square(0.5625 * delta_t)
    )

# pythran export haversine(float[:], float[:])
def haversine(x, y):
    lon1, lat1 = x[0] * np.pi/180, x[1] * np.pi/180
    lon2, lat2 = y[0] * np.pi/180, y[1] * np.pi/180
    delta_t = x[2]-y[2]
    return -np.sqrt(
        np.square(.1 * haversine_distance(lat1, lat2, lon1, lon2))
        + np.square(0.5625 * delta_t)
    )