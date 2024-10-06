from sklearn.model_selection import train_test_split
import pandas as pd

# Load the classification data set
d = pd.read_excel('output/all1.xlsx')
data=pd.DataFrame(d)


# Specify the features of interest and the classes of the target
features = data.keys()
classes = data.iloc[:,-1]

# Extract the instances and target
X = data[features[:-1]]
y = classes

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
bayes = SVC()
visualizer = ClassificationReport(bayes, classes=[1,2], support=True,)

visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()