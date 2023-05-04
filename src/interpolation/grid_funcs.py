import numpy as np

# pythran export sign_add(int, int)
# pythran export sign_add(float, float)
# pythran export sign_add(float, int)
# pythran export sign_add(int, float)
def sign_add(x, y):
    if x <= 0:
        return x + y 
    return x - y

# pythran export _landmask_coord_bool(float[:], int)
# pythran export _landmask_coord_bool(float[:], float)
def _landmask_coord_bool(landmask, value):
    bool_arr =  np.where(landmask > value)[0]
    if len(bool_arr) == 0:
        if landmask[0] > value:
            return 0
        return len(landmask)
    return bool_arr[0]

# pythran export block_mean_loop(int, int, float, float, float, float[:], float[:], float64[:])
# pythran export block_mean_loop(int, int, int, float, float, float[:], float[:], float64[:])
# pythran export block_mean_loop(int, int, float, float, float, float[:], float[:], float32[:])
# pythran export block_mean_loop(int, int, int, float, float, float[:], float[:], float32[:])
def block_mean_loop(x_size,y_size,resolution,start_pos_x,start_pos_y,data_lon,data_lat,vals):
    block_grid = np.zeros((y_size,x_size))
    sizes = np.zeros((y_size,x_size))
    lons = start_pos_x + np.arange(0,x_size)*resolution
    lats = start_pos_y + np.arange(0,y_size)*resolution

    for val,lon,lat in zip(vals,data_lon,data_lat):
        idxlat = np.where((lat < lats) & (lat+resolution > lats))[0]
        idxlon = np.where((lon < lons) & (lon+resolution > lons))[0]

        for i in idxlat:
            for j in idxlon:
                block_grid[i,j] += val
                sizes[i,j] += 1
    block_grid /= sizes
    return block_grid

# pythran export block_mean_loop_time(int, int, int, float, int64[:], float, float, int64[:], float64[:], float64[:], int64[:], float64[:,:] or float32[:,:])
def block_mean_loop_time(
    x_size,
    y_size,
    t_size,
    s_res,
    t_res,
    start_pos_x,
    start_pos_y,
    start_pos_t,
    data_lon,
    data_lat,
    data_time,
    vals
):
    lons = start_pos_x + np.arange(0, x_size + 1) * s_res
    lats = start_pos_y + np.arange(0, y_size + 1) * s_res
    times = start_pos_t + np.arange(0, t_size + 1) * t_res

    block_grid = np.zeros((len(vals), 4 + vals.shape[1]), dtype=np.float64)

    count = 0
    lookup = {}
    for val, lon, lat, time in zip(vals, data_lon, data_lat, data_time):
        idxlat = np.where(
            (lat < lats + s_res/2) & (lat >= lats - s_res/2)
        )[0]
        idxlon = np.where(
            (lon < lons + s_res/2) & (lon >= lons - s_res/2)
        )[0]
        idxtime = np.where(
            (time < times + t_res/2) & (time >= times - t_res/2)
        )[0]
        
        for i in idxlon:
            for j in idxlat:
                for t in idxtime:
                    grididx = (i * (x_size + 1) + j) * (t_size + 1) + t
                    if grididx in lookup:
                        tmpidx = lookup.get(grididx, 0)
                        block_grid[tmpidx, 0] += 1
                        block_grid[tmpidx, 4:] += val
                    else:
                        lookup[grididx] = count
                        block_grid[count, 0] = 1
                        block_grid[count, 1] = lons[i]
                        block_grid[count, 2] = lats[j]
                        block_grid[count, 3] = times[t]
                        block_grid[count, 4:] = val
                        count += 1
    block_grid = block_grid[:count]
    block_grid[:, 4:] = block_grid[:, 4:] / block_grid[:, 0:1]
    return block_grid[:, 1:]

