
import dtaidistance as dis
import pandas as pd
import array as arr


# path = pd.read_excel("datasets\Kgk .xls")

# ----------------------------------------------
def getDtwResult(path):
    dataSet = pd.read_excel(path)

    DTW_result = {}
    data = pd.DataFrame(dataSet).values.T

    for i in range(len(data)):
        for j in range(i, len(data)):
            if j == i:
                continue
            DTW_result[dis.dtw.distance_fast(arr.array('d', data[i]), arr.array('d', data[j]))] = (i, j)
    return DTW_result








