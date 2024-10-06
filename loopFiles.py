
import os
import dtaidistance as dis
import array as arr
import glob
import pandas as pd

os.chdir('datasets\class2')
fileList = glob.glob('*.xls')
dtw_for_every_file = {}
for file in fileList:
    data0 = pd.read_excel(file)
    data = pd.DataFrame(data0)

    DTW_result = []

    for i in data:
        for j in data:
            if j == i:
                continue
            DTW_result.append(dis.dtw.distance_fast(arr.array('d', data[i]), arr.array('d', data[j])))

    dtw_for_every_file[str(file.title())] = DTW_result

print(dtw_for_every_file)

pd.DataFrame(dtw_for_every_file).to_csv('class2_result.csv', index=False)
