import numpy as np
import dtaidistance as dis
import pandas as pd


class dtw_process:

    def apply_dtw_algorthim_on_data(self,dataset):
        DTW_result = {}
        data=dataset
        for i in data:
            for j in data:
                if j == i:
                    continue
                DTW_result[np.around(dis.dtw.distance(data[i], data[j]), decimals=3)] = (i, j)

        return DTW_result





