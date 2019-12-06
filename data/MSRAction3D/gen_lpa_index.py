import numpy as np


line_index = np.array([[20, 3], [3, 1], [3, 4], [3, 2], [1, 8],  # for select ll angle
                       [8, 10], [10, 12], [4, 7], [7, 5], [5, 14],
                       [14, 16], [16, 18], [7, 6], [6, 15], [15, 17],
                       [17, 19], [2, 9], [9, 11], [11, 13], [20, 12],
                       [12, 18], [18, 19], [19, 13], [13, 20], [12, 13],
                       [13, 18], [18, 20], [20, 19], [19, 12], [13, 9],
                       [12, 8], [20, 4], [18, 14], [19, 15]])

point_index = [12, 10, 8, 1, 1, 3, 2, 2, 9, 11, 13, 20, 3, 4, 7, 7,
               5, 6, 5, 14, 16, 18, 6, 15, 17, 19]

len_line = len(line_index)
index = np.zeros((897, 3), dtype=np.int32)
cnt = 0
for i in range(1, 21):
    for j in range(len_line):
        if i != line_index[j, 0] and i != line_index[j, 1]:
            index[cnt, 0] = i
            index[cnt, 1] = line_index[j, 0]
            index[cnt, 2] = line_index[j, 1]
            cnt = cnt + 1

for i in range(cnt):
    print('[%d,%d,%d],' % (index[i, 0], index[i, 1], index[i, 2]))
