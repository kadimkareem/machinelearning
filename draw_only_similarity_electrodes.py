import numpy as np
import dtaidistance as dis
import pandas as pd
import networkx as nt
import matplotlib.pyplot as plt
import array as arr
import statistics as sts

DTW_result = {}
DTW_result2= {}
data = pd.read_excel('datasets/congurent_subject1.xls')
data = pd.DataFrame(data)

# ----------------------------------------------
for i in data:
    for j in data:
        if j == i:
            continue
        DTW_result[np.around(dis.dtw.distance_fast(arr.array('d', data[i]), arr.array('d', data[j])))] = (i, j)
dataInfo=pd.DataFrame(DTW_result)
meanValue=sts.mean(DTW_result.keys())


new={i:DTW_result[i] for i in DTW_result if i>meanValue}
print(new)

edgeslabel = dict([[v, k] for k, v in new.items()])
electrodes = []
# print(data_matrix)
for i in data:
    electrodes.append(i)
# --------------------------------------------------plot---
g = nt.Graph()
edgeColor = [i for i in range(len(edgeslabel.values()))]
edges = edgeslabel.keys()
plt.figure(figsize=(20, 20))
# edge_width=[ i/10.100 for i in edgs.keys()]
edge_width = 3
g.add_nodes_from(electrodes)

pos = {'F3': (0.67303, 0.54501), 'FP1': (0.95106, -0.30902), 'FP2': (0.95106, 0.30902), 'P3': (-0.67303, 0.54501),
       'O1': (-0.95106, 0.30902), 'C3': (4.3298e-017, 0.70711), 'FZ': (0.70711, 0), 'T7': (6.1232e-017, 1),
       'CZ': (6.1232e-017, 0), 'F4': (0.67303, -0.54501), 'C4': (4.3298e-017, -0.70711), 'T8': (6.1232e-017, -1),
       'PZ': (-0.70711, -8.6596e-017), 'P4': (-0.67303, -0.54501), 'O2': (-0.95106, -0.30902), 'OZ': (-1, -1.2246e-017)}
# g.add_edges_from(edgs.values())
nt.draw(g, pos,with_labels=1,node_size=3000,node_color='r')
nt.draw_networkx_edges(g,pos,edgelist=edges,width=3,edge_color=edgeColor,edge_cmap=plt.cm.viridis,nodelist=g.nodes)
nt.draw_networkx_edge_labels(g, pos, edge_labels=edgeslabel, font_color='red',alpha=0.5,label_pos=0.3)
# nt.draw_networkx_edges(g,pos,edgs.values(),width=edge_width,edge_color=edgeColor,cmap='hot')
fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
fig.savefig('draw_only_similarity_electrodes_for_congruent_subject.png', dpi=100)
plt.show()

