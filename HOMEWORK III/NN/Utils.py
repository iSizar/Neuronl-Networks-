import numpy as np


def crate_batches(batch_count: int, all_data: np.array) -> list:
    ret_list = []
    idx = 1
    batch_size = batch_count
    while (idx * batch_size) <= len(all_data):
        ret_list.append(all_data[(idx-1)*batch_size: min(idx*batch_size, len(all_data))])
        idx += 1
    return ret_list