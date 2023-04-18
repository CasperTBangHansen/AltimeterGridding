import numpy as np

# pythran export group_by_time(int64[:], int64[:])
def group_by_time(times, crossover_times):
    i = 0
    max_len = len(crossover_times) - 1
    
    indexing = np.zeros(len(times), dtype=np.int64)
    current_time = crossover_times[i]
    for j, time in enumerate(times):
        while time > current_time:
            i += 1
            if i > max_len:
                indexing[j:] = -1
                break
            current_time = crossover_times[i]
        if i > max_len:
            break
        indexing[j] = i
    return indexing