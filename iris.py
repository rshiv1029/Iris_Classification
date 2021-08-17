import sys
import scipy as sp
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# dimensions
dimensions = dataset.shape
# peek
print(dataset.head(20))
# summary
print(dataset.describe())

# Data Visualization 
# We want to see what the data looks like so we can 
# choose a proper model

# Univariate Plot
# We want to see how each input variable is distributed

# Box & Whisker Plot 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# Histograms
dataset.hist()
pyplot.show()

# Multivariate Plot
scatter_matrix(dataset)
pyplot.show()

# Split our dataset into train and test
# Set 80% of data to be training and 20% for testing
array = dataset.values
X=dataset.drop('class', axis=1)
y=dataset['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)

# Scale our data to be a unit distribution
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now that we split our test and train data we will apply our 
# classification models
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train,y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test,y_test)
print(score)