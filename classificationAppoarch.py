import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# Import train_test_split function



data=pd.read_excel('output/all1.xlsx')
data=pd.DataFrame(data)


#select feature and class label
features = data.keys()
classlabel = data.iloc[:,-1]
X = data[features[:-1]]
Y = classlabel
classes=[1,2]



# class distribution
print(data.groupby(1.0).size())
# xx=features[1]
# yy=features[2]
# data.plot(kind='scatter',x=xx,y=yy, colormap='viri')
# plt.show()


# ploting dataset distrution
# rslt={}
# rslt=data.groupby(1.0).size()
#
# print(rslt
#       )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test
# print (X_train.shape, y_train.shape)
# print (X_test.shape, y_test.shape)
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

# Split-out validation dataset
validation_size = 0.20
seed = 7
scoring = 'accuracy'


# evaluate each model in turn
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', GaussianNB()))

results = []
names = []


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Spot Check Algorithms
for name, model in models:
      kfold = model_selection.KFold(n_splits=10, random_state=seed)
      cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      model.fit(X_train, Y_train)
      if name is 'KNN':
            model.n_neighbors=1
      y_pred = model.predict(X_validation)
      model.predict(X_validation)
      results.append(name)
      results.append('cv_results:')
      results.append(cv_results)
      results.append('confusion_matrix:')
      results.append(confusion_matrix(Y_train,y_pred))
      results.append('classification_report:')
      results.append(classification_report(Y_train,y_pred))
      results.append('accuracy_score:')
      results.append(accuracy_score(Y_train, y_pred))
      results.append('---------------------------------------')
      results.append('/n')
      names.append(name)
      # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std(),'report',results)



for i in results:
    print(i,flush=True)