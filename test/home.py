
import dtaidistance as dis
import pandas as pd
import array as arr

DTW_result = {}
dataset = pd.read_excel('datasets/1scnd.xls')
data = pd.DataFrame(dataset)

# ----------------------------------------------
for i in range(len(data)):
    for j in range(i, len(data)):
        if j == i:
            continue
        DTW_result[dis.dtw.distance_fast(arr.array('d',data[i]), arr.array('d',data[j]))] = (i, j)

print(DTW_result)