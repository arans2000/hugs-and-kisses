

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

#lets pick some algorithms to utilise
# Split-out validation dataset


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
#credit to machinelearningmastery.com for this model list
#we'll just go with LR and SVM for now

#testing the effectivenesss of the machines on the training set
results = []
names = []
print('Testing models:')
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, trainsetx, trainsety, cv=kfold, scoring='f1')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

print('end test \n \n')
#scoring = accuracy results:
#SVM seems most effective but become exponentially long to train with dataset taking two minutes to reach a 53~% accuracy
#NB takes significantly less time and is second most accurate reaching 51.4% accuracy with the same 1/10 slice of the training set
#however larger subsets of the training data do not substantially increase this score
#results from running 1/10 of training set
#LR: 0.498900 (0.016885)
#LDA: 0.499300 (0.017855)
#KNN: 0.506900 (0.018097)
#CART: 0.497400 (0.021086)
#NB: 0.514000 (0.014839)
#SVM: 0.533000 (0.011593)


#scoring = f1 results:
#higher results across the board particularly for logistic regression and linear discriminant analysis
#LR: 0.562779 (0.019180)
#LDA: 0.562353 (0.019817)
#KNN: 0.510740 (0.017884)
#CART: 0.505249 (0.020554)
#NB: 0.511954 (0.018873)
#SVM: 0.551646 (0.012082)
#running full set through  LR and LDA increased f1 scores to 66%
#LR: 0.664890 (0.004952)
#LDA: 0.664872 (0.004944)

#ran 1/2 set of training set for all:
#LR: 0.656386 (0.011779)
#LDA: 0.656199 (0.011819)
#KNN: 0.525533 (0.006223)
#CART: 0.515621 (0.007442)
#NB: 0.536666 (0.012270)
#SVM: 0.583134 (0.009338)
#LR adn LDA came out the clear winners, and SVM simply took too long to process to reasonably justify it, so I'll settle for LR and LDA

#They will hand out all kinds of dangerous mind altering drugs for basically no money. Heres the thing, you can tell by the people in the chat making fun of the fact that im clearly balding, ive looked into this obviously and there is a medication that is known to be quite good that you can self-prescribe. You can literally ask for it and theyll give it to you, you dont even have to seee a doctor for it, and the warning label on the side "this may make you want to kill yourself". Like, theyll hand out depression giving drugs if it might give you a little bit more hair for longer. But if you desperately want to alleviate gender dysphoria, theyre like "yea were gonna need half a decade of nonsense first". Like, fuck off. 

# train machine

trainfraction = 1 
trainset = trainarr.iloc[::trainfraction,:]
trainsetx = trainarr.iloc[::trainfraction,:10]
trainsety = trainarr.iloc[::trainfraction,10]
print('training LR and LDA')
print('predicting with LR and LDA \n \n ')
LR = LogisticRegression(solver='liblinear', multi_class='ovr')
LDA = LinearDiscriminantAnalysis()


LR.fit(trainsetx, trainsety)
LRpredictions = LR.predict(validatex)
print('Logistic Regression prediction analysis:')
print(accuracy_score(validatey, LRpredictions))
print(confusion_matrix(validatey, LRpredictions))
print(classification_report(validatey, LRpredictions))


LDA.fit(trainsetx, trainsety)
LDApredictions = LDA.predict(validatex)
print('\n \n \'Logistic discriminant analysis\' prediction analysis:')
print(accuracy_score(validatey, LDApredictions))
print(confusion_matrix(validatey, LDApredictions))
print(classification_report(validatey, LDApredictions))


#The model result analysis isnt too promising, but the proof is in the scatterplot!



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
	ax.scatter(x, y, alpha=0.1, c=color, edgecolors='none', s=30, label=group)

plt.title(title)
plt.legend(loc=2)
plt.show()




#graphing prediction set



pZ = [[]] * 5 #initialise list of 'zero' rows
pO = [[]] * 5 #initialise list of 'one' rows


#populate our lists with values based on 'zero'/'one' rows and which columns they occupied
for X, Y, ind in zip(evensmpleout, oddsmpleout, LRpredictions):
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



title = "LR predicted values: zeros vs ones \n LDA predicts a disproportionate amount of ones to zeros"

One = (npO[:,::5,0],npO[:,::5,1])
Zero = (npZ[:,:,0],npZ[:,:,1])



data = (Zero, One)
colors = ("blue", "orange")
groups = ("0", "1")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

for data, color, group in zip(data, colors, groups):
	x, y = data
	ax.scatter(x, y, alpha=0.25, c=color, edgecolors='none', s=30, label=group)

plt.title(title)
plt.legend(loc=2)
plt.show()


