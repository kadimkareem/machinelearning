
import os
import dtaidistance as dis
import array as arr
import glob
import pandas as pd

os.chdir('Stroop_datasets/new_class2_incongruent')
fileList = glob.glob('*.xls')
dtw_for_every_file = {}
for file in fileList:
    data_set = pd.read_excel(file)
    data = pd.DataFrame(data_set).values.T

    DTW_result = []

    for i in range(len(data)):
        for j in range(i,len(data)):
            if j == i:
                continue
            DTW_result.append(dis.dtw.distance_fast(arr.array('d', data[i]), arr.array('d', data[j])))

    dtw_for_every_file[str(file.title())] = DTW_result

print(len(dtw_for_every_file.values()))

pd.DataFrame(dtw_for_every_file).to_csv('class22_congruent.csv', index=False)
