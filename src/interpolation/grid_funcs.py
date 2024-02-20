import numpy as np
import itertools

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
                        block_grid[tmpidx, 1] += lon # new
                        block_grid[tmpidx, 2] += lat # new
                        block_grid[tmpidx, 4:] += val
                    else:
                        lookup[grididx] = count
                        block_grid[count, 0] = 1
                        # block_grid[count, 1] = lons[i]
                        # block_grid[count, 2] = lats[j]
                        block_grid[count, 1] = lon # new
                        block_grid[count, 2] = lat # new
                        block_grid[count, 3] = times[t]
                        block_grid[count, 4:] = val
                        count += 1
    block_grid = block_grid[:count]
    block_grid[:, 1:3] = block_grid[:, 1:3] / block_grid[:, 0:1]
    block_grid[:, 4:] = block_grid[:, 4:] / block_grid[:, 0:1]
    return block_grid[:, 1:]

# pythran export nlogn_median(float32[:] or float64[:])
def nlogn_median(l):
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[int((len(l)-1) / 2)]
    else:
        return 0.5 * (l[int(len(l) / 2 - 1)] + l[int(len(l) / 2)])

# pythran export block_median_loop_time(int, int, int, float, int64[:], float, float, int64[:], float64[:], float64[:], int64[:], float64[:,:] or float32[:,:])
def block_median_loop_time(
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

    block_grid = []
    for _ in range(4+vals.shape[1]):
        block_grid.append([])

    for k, variable in enumerate(vals.T):
        var_list = []
        count = 0
        lookup = {}
        for val, lon, lat, time in zip(variable, data_lon, data_lat, data_time):
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
                            block_grid[0][tmpidx] += 1
                            var_list[tmpidx].append(val)
                        else:
                            lookup[grididx] = count
                            if len(block_grid[1]) == len(var_list):
                                block_grid[0].append(1)
                                block_grid[1].append(lons[i])
                                block_grid[2].append(lats[j])
                                block_grid[3].append(times[t])
                            var_list.append([val])
                            count += 1
        medians = []
        for l in var_list:
            medians.append(nlogn_median(l))
        block_grid[4+k] = medians
    return np.array(block_grid)[1:].T

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

# pythran export segment_grid(float32[::,:] or float64[::,:], float32[:,:] or float64[:,:], int)
def segment_grid(block_grid, interp_coords, k):
    segmented_block_grid = []
    for int_lon, int_lat in zip(interp_coords[:,0], interp_coords[:,1]):
        distance = haversine_distance(int_lat, block_grid[:, 1], int_lon, block_grid[:, 0])
        sort_idx = np.argsort(distance)
        if not isinstance(sort_idx, int):
            sort_idx_k = sort_idx[:k]
            block_grid_lon=block_grid[:,0][sort_idx_k]
            block_grid_lat=block_grid[:,1][sort_idx_k]
        else:
            block_grid_lon=block_grid[sort_idx[0],0]
            block_grid_lat=block_grid[sort_idx[0],1]
        segmented_block_grid.append(
            [
                block_grid_lon, 
                block_grid_lat, 
                np.ones(k, dtype=np.float64)*interp_coords[0,2]]
             )
    return segmented_block_grid


