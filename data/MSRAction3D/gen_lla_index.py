import numpy as np
index = np.array([[20, 3], [3, 1], [3, 4], [3, 2], [1, 8],  # for select ll angle
                  [8, 10], [10, 12], [4, 7], [7, 5], [5, 14],
                  [14, 16], [16, 18], [7, 6], [6, 15], [15, 17],
                  [17, 19], [2, 9], [9, 11], [11, 13], [20, 12],
                  [12, 18], [18, 19], [19, 13], [13, 20], [12, 13],
                  [13, 18], [18, 20], [20, 19], [19, 12], [13, 9],
                  [12, 8], [20, 4], [18, 14], [19, 15]])
n_index = index.shape[0]
index_out = np.zeros((741, 4), dtype=np.int32)
n = 0
for i in range(n_index-1):
    for j in range(i+1, n_index):
        index_out[n, 0] = index[i, 0]
        index_out[n, 1] = index[i, 1]
        index_out[n, 2] = index[j, 0]
        index_out[n, 3] = index[j, 1]
        n = n + 1

for i in range(n):
    print('[%d,%d,%d,%d],' % (index_out[i, 0],
                              index_out[i, 1],
                              index_out[i, 2],
                              index_out[i, 3]))
