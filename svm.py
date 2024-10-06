from sklearn import svm
import pandas as pd
from yellowbrick.classifier import ClassificationReport , ClassPredictionError
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# Import train_test_split function
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores


data=pd.read_excel('output/all1.xlsx')
data=pd.DataFrame(data)
#select feature and class label
features = data.keys()
classlabel = data.iloc[:,-1]
# Split dataset into training set and test set
X = data[features[:-1]]
y = classlabel
classes=[1,2]
# class distribution
print(data.groupby(1.0).size())

# ploting dataset distrution
# rslt={}
# rslt=data.groupby(1.0).size()
#
# print(rslt
#       )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test
# clf = svm.SVC(kernel='linear') # Linear Kernel
#
# #Train the model using the training sets
# clf.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
# print("confusion_matrix:",confusion_matrix(y_test,y_pred))
# print("classification_report:",classification_report(y_test,y_pred))
# # Model Accuracy: how often is the classifier correct?
# print("Accuracy:",accuracy_score(y_test, y_pred))
# # print("Accuracy:",accuracy_score(y_test, y_pred))
validation_size = 0.20
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
#
# for name, model in models:
#       kfold = model_selection.KFold(n_splits=10, random_state=seed)
#       cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
#       model.fit(X_train, y_train)
#       if name is 'KNN':
#             model.n_neighbors=10
#       y_pred = model.predict(X_test)
#       model.predict(X_test)
#       results.append(name)
#       results.append('cv_results:')
#       results.append(cv_results)
#       results.append('confusion_matrix:')
#       results.append(confusion_matrix(y_test,y_pred))
#       results.append('classification_report:')
#       results.append(classification_report(y_test,y_pred))
#       results.append('accuracy_score:')
#       results.append(accuracy_score(y_test, y_pred))
#       results.append('---------------------------------------')
#       names.append(name)
#       # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std(),'report',results)
#       # print(results)


#
# Instantiate the classification model and visualizer

#prediction error ploting
# for name, model in models:
#       # Instantiate the classification model and visualizer
#       visualizer = ClassPredictionError(model, classes=classes)
#
#       visualizer.fit(X_train, y_train)
#       # Evaluate the model on the test data
#       visualizer.score(X_test, y_test)
#
#       # Draw visualization
#       g = visualizer.poof()

#confusion matrix plot
for name, model in models:


      # The ConfusionMatrix visualizer taxes a model
      cm = ConfusionMatrix(model, classes=classes ,percent=True)

      # Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
      cm.fit(X_train, y_train)


      # To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
      # and then creates the confusion_matrix from scikit-learn.
      cm.score(X_test, y_test)

      # How did we do?
      cm.poof()
      print()

#----------------------------------
#cross validation score
# Create a new figure and axes
#
# for name,model in models:
#       _, ax = plt.subplots()
#
#       # Create a cross-validation strategy
#       cv = StratifiedKFold(10)
#
#       # Create the cv score visualizer
#       oz = CVScores(
#             model, ax=ax, cv=cv, scoring='f1_weighted'
#       )
#
#       oz.fit(X, y)
#       oz.poof()
#