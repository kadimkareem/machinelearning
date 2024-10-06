
import numpy as np
import dtaidistance as dis
import pandas as pd
import networkx as nt
import matplotlib.pyplot as plt

DTW_result = {}
data = pd.read_excel("Kgk .xls").head(3)
data=pd.DataFrame(data)

# ----------------------------------------------
for i in data:
    for j in data:
        if j==i:
            continue
        DTW_result[np.around(dis.dtw.distance(data[i], data[j]), decimals=3)] = (i, j)

# ----------------------------------------------------------------
G = nt.Graph()
G.add_nodes_from(DTW_result.keys())
# draw---------------------------------------------------------
nt.draw_networkx(G,nt.random_layout(G),with_labels=1,node_color=range(0,len(G.nodes)),cmap='viridis_r')

plt.show()