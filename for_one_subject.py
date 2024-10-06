
import os
import dtaidistance as dis
import array as arr
import glob
import pandas as pd
os.chdir('datasets/class2')
fileList=glob.glob('*.xls')
dtw_for_every_file={}

for file in fileList:

    data=pd.read_excel(file).values.T
    DTW_result = []
    count = len(data)
    j=7
    for i in range(15):  # 8

            DTW_result.append(dis.dtw.distance_fast(arr.array('d', data[j]), arr.array('d', data[i])))

    dtw_for_every_file[str(file.title())] = DTW_result



print(len(DTW_result))
print(len(dtw_for_every_file.values()))
print(len(dtw_for_every_file.keys()))


pd.DataFrame(dtw_for_every_file).to_csv('class2_result213.csv', index=False)
