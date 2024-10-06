
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

datafile='datasets/incoungerunet_subject1.xls'
data = pd.read_excel(datafile)
df=pd.DataFrame(data)

df.plot(subplots=True ,title='incongruent status')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('incounguren_sepertaed_columns.png', dpi=100)
plt.show()





