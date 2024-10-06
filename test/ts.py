
import dtaidistance as dis
import pandas as pd
import array as arr

DTW_result = {}
data = pd.read_excel("datasets\Kgk .xls").head(500)
data=pd.DataFrame(data)

# ----------------------------------------------
for i in range(len(data)):
    for j in range(0,i):
        if j==i:

            continue
        print(data[i],data[j],end=' ',flush=True)

print('')







