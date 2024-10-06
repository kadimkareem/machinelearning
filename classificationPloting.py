import numpy as np
import pandas
from yellowbrick.classifier import ClassificationReport, ClassPredictionError
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Import train_test_split function
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores

# load dataset
dataset = pandas.read_excel('output/all1.xlsx')
data = pandas.DataFrame(dataset)

# shape
print(dataset.shape)

# descriptions
print(dataset.describe())

# class distribution
rslt = {}
rslt = data.groupby(1.0).size()
print(rslt)

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
Y = classlabel

# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
# models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', gamma='auto')))
# evaluate each model in turn
results = []
names = []
classes = [1, 2]
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

for name, model in models:
    # names.append(name)
    # kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_validation)
    # report
    # visualizer = ClassificationReport(model, classes=classes, support=True)
    # visualizer.fit(X_train, Y_train)  # Fit the visualizer and the model
    # visualizer.score(X_validation, Y_validation)  # Evaluate the model on the test data
    # g = visualizer.poof()
    # confusion matrinx
    plot_confusion_matrix(Y_validation, y_predict, classes=classes,
                          title='Confusion matrix of {}'.format(name))
    plt.show()


