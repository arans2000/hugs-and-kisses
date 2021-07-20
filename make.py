# a.txt , test.txt and train.txt are all arrays, 
# when the test.txt array is multiplies by the inverse of the a.txt array it should form a pattern based on the machine predicted binary values of its rows

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import matplotlib.pyplot as plt
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from cycler import cycler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


aarray = read_csv('a.txt', sep=" ", names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
ainverse = numpy.linalg.inv(aarray)

# Load training dataset (X)
trainarr = read_csv('train.txt', sep=" ", names=['tra', 'trb', 'trc', 'trd', 'tre', 'trf', 'trg', 'trh', 'tri', 'trj', 'trout'])

trainfraction = 25 
trainset = trainarr.iloc[::trainfraction,:]
#print(trainset.head(20))
#print(trainset.describe())
#print(trainset.groupby('trout').size())


#grab the first 10 columns of every 50th (fraction of training set, may increase later) row
#x
trainsetx = trainarr.iloc[::trainfraction,:10]

#grab the 11th column of the training set to keep track of which row is a 'zero' and which is 'one'
#y
trainsety = trainarr.iloc[::trainfraction,10]

#we can run the visualisation of this sample set with the same steps from line 'N'

#load in the test data 
#x
validatex = read_csv('test.txt', sep=" ", names=['tra', 'trb', 'trc', 'trd', 'tre', 'trf', 'trg', 'trh', 'tri', 'trj'])
#y
validatey = read_csv('key.txt', sep=" ", names=['trout'])

#try out a few models


#model =  LogisticRegression(solver='liblinear', multi_class='ovr')
#model =  LinearDiscriminantAnalysis()
#model =  KNeighborsClassifier()
#model =  DecisionTreeClassifier()
#model =  GaussianNB()
#model = SVC(gamma='auto')
model = SVC( gamma=0.4)
#divvy up the train set for SVC
trainfraction = 5 
trainset = trainarr.iloc[::trainfraction,:]
trainsetx = trainarr.iloc[::trainfraction,:10]
trainsety = trainarr.iloc[::trainfraction,10]
print('training {}'.format(model))
print('predicting with {} \n \n '.format(model))



model.fit(trainsetx, trainsety)
modelpredictions = model.predict(validatex)
print('\'{}\' analysis:'.format(model))
print(accuracy_score(validatey, modelpredictions))
print(confusion_matrix(validatey, modelpredictions))
print(classification_report(validatey, modelpredictions))


#graphing the key set


index = validatey.to_numpy()
#graphing the predicitions
nptest = validatex.to_numpy()
#index = LDApredictions.to_numpy()
testout = nptest.dot(ainverse)
#split it into odd and even elements to compare x0 v x1, x2 v x3, x4 v x5, etc.
evensmpleout=testout[:,1::2]#all even
oddsmpleout=testout[:,0::2] #all odd


kZ = [[]] * 5 #initialise list of 'zero' rows
kO = [[]] * 5 #initialise list of 'one' rows


#populate our lists with values based on 'zero'/'one' rows and which columns they occupied
for X, Y, ind in zip(evensmpleout, oddsmpleout, index):
	if ind == 0:
		kZ[0].append([X[0],Y[0]])
		kZ[1].append([X[1],Y[1]])
		kZ[2].append([X[2],Y[2]])
		kZ[3].append([X[3],Y[3]])
		kZ[4].append([X[4],Y[4]])
		
	else:
		kO[0].append([X[0],Y[0]])
		kO[1].append([X[1],Y[1]])
		kO[2].append([X[2],Y[2]])
		kO[3].append([X[3],Y[3]])
		kO[4].append([X[4],Y[4]])


#change to arrays for slicing
npZ = numpy.array(kZ)
npO = numpy.array(kO)



title = "Key set values: zeros vs ones"

One = (npO[:,:,0],npO[:,:,1])
Zero = (npZ[:,:,0],npZ[:,:,1])



data = (Zero, One)
colors = ("blue", "orange")
groups = ("0", "1")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

for data, color, group in zip(data, colors, groups):
	x, y = data
	ax.scatter(x, y, alpha=0.05, c=color, edgecolors='none', s=30, label=group)

plt.title(title)
plt.legend(loc=2)
plt.show()




#graphing prediction set



pZ = [[]] * 5 #initialise list of 'zero' rows
pO = [[]] * 5 #initialise list of 'one' rows


#populate our lists with values based on 'zero'/'one' rows and which columns they occupied
for X, Y, ind in zip(evensmpleout, oddsmpleout, modelpredictions):
	if ind == 0:
		pZ[0].append([X[0],Y[0]])
		pZ[1].append([X[1],Y[1]])
		pZ[2].append([X[2],Y[2]])
		pZ[3].append([X[3],Y[3]])
		pZ[4].append([X[4],Y[4]])
		
	else:
		pO[0].append([X[0],Y[0]])
		pO[1].append([X[1],Y[1]])
		pO[2].append([X[2],Y[2]])
		pO[3].append([X[3],Y[3]])
		pO[4].append([X[4],Y[4]])


#change to arrays for slicing
npZ = numpy.array(pZ)
npO = numpy.array(pO)



title = "{} predicted values: zeros vs ones".format(model)

One = (npO[:,:,0],npO[:,:,1])
Zero = (npZ[:,:,0],npZ[:,:,1])



data = (Zero, One)
colors = ("blue", "orange")
groups = ("0", "1")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

for data, color, group in zip(data, colors, groups):
	x, y = data
	ax.scatter(x, y, alpha=0.05, c=color, edgecolors='none', s=30, label=group)

plt.title(title)
plt.legend(loc=2)
plt.show()

#note: decision tree classifier assigns a more accurate distribution of ones to zeros then LDA, LR or KNN.
# naieve bays accomplishes similar results
#
print("predicted zeros and ones, \n zeros:")
print(len(pO[0]))
print("ones:")
print(len(pZ[0]))
#Will settle on SVC
print("creating {}...".format(sys.argv[1]))

with open(sys.argv[1], 'w') as f:
    for item in modelpredictions:
        f.write("{}\n".format(item))
