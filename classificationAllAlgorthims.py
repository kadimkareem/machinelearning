# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC


#load dataset
dataset = pandas.read_csv('output/all1csv.csv')
data=pandas.DataFrame(dataset)

# shape
print(dataset.shape)





# descriptions
print(dataset.describe())



# class distribution
rslt={}
# rslt=data.groupby(1.0).size()
# print(rslt)


# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(12,12), sharex=False, sharey=False)
# plt.show()



# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(12,12), sharex=False, sharey=False)
# plt.show()




# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()






features = [i for i in data.keys()]
classlabel = data.iloc[:, -1]

# Extract the instances and target
X = data[features[:-1]]
X=np.array(X,dtype='f')
Y= np.array(classlabel,dtype='f')

# Split-out validation dataset
validation_size = 0.20
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)




# Test options and evaluation metric
seed = 7
scoring = 'accuracy'



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
# models.append(('linear regression', LinearRegression()))
models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf',gamma='auto')))
#evaluate each model in turn
results = []
names = []
for name, model in models:
	names.append(name)
	# kfold = model_selection.KFold(n_splits=10, random_state=seed)
	# cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	model.fit(X_train, Y_train)
	y_predict = model.predict(X_validation)
	results.append(name)
	# results.append(cv_results)
	results.append(accuracy_score(Y_validation, y_predict))
	# results.append(accuracy_score(Y_validation, y_predict.round(),normalize=False))
	results.append(classification_report(Y_validation, y_predict))
	results.append(confusion_matrix(Y_validation,y_predict))
	# msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	# results.append(msg)
	results.append('-----------------------')


#prein reaults
for i in results:
	print(i,flush=True)

# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

#
#
#
# # Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))