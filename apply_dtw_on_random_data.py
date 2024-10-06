
import numpy as np
import dtaidistance as dis
import pandas as pd

DTW_result = {}
data=np.random.rand(16,16)
random_data=pd.DataFrame(data).values.T
# ----------------------------------------------

for i in range(len(random_data)):
    for j in range(i, len(random_data)):
        if j == i:
            continue
        DTW_result[np.around(dis.dtw.distance(random_data[i], random_data[j]), decimals=3)] = (i, j)

result = dict([[v,k] for k,v in DTW_result.items()])
print(result)






