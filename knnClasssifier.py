# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection._validation import cross_val_predict

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.plotting import parallel_coordinates


#load dataset
knn=KNeighborsClassifier()
dataset = pandas.read_excel('output/all1.xlsx')
df=pd.DataFrame(dataset)
classss=df.iloc[:,-1]
X=df[:-1]
y=np.array([classss])

predicted = cross_val_predict(knn, X, y, cv=10)

