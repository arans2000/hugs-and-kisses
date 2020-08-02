
#exec(open("b.py").read())

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
#convert numpyarray to dataframe
#dfainverse = pandas.DataFrame({'a': ainverse[:, 0], 'b':ainverse[:, 1], 'c':ainverse[:, 2], 'd':ainverse[:, 3], 'e':ainverse[:, 4], 'f':ainverse[:, 5], 'g':ainverse[:, 6], 'h':ainverse[:, 7], 'i':ainverse[:, 8], 'j':ainverse[:, 9]})
# Load training dataset
trainarr = read_csv('train.txt', sep=" ", names=['tra', 'trb', 'trc', 'trd', 'tre', 'trf', 'trg', 'trh', 'tri', 'trj', 'trout'])



#take the first 10 columns of  row and convert it to numpy array (using the first 5000 rows for testing)
#here we use the first 5000 rows
#smplesize = 5000
#nptrainarrsmpl = trainarr.iloc[:smplesize,:10].to_numpy()
##grab the 11th column of the training set to keep track of which row is a 'zero' and which is 'one'
#index = trainarr.iloc[:smplesize,10].to_numpy()


#here we use every 50th row
smplesize = 50
nptrainarrsmpl = trainarr.iloc[::smplesize,:10].to_numpy()
#grab the 11th column of the training set to keep track of which row is a 'zero' and which is 'one'
index = trainarr.iloc[::smplesize,10].to_numpy()




#multiply that by the inverse of 'a'
smpleout = nptrainarrsmpl.dot(ainverse)
#split it into odd and even elements to compare x0 v x1, x2 v x3, x4 v x5, etc.
evensmpleout=smpleout[:,1::2]#all even
oddsmpleout=smpleout[:,0::2] #all odd


Z = [[]] * 5 #initialise list of 'zero' rows
O = [[]] * 5 #initialise list of 'one' rows


#populate our lists with values based on 'zero'/'one' rows and which columns they occupied
for X, Y, ind in zip(evensmpleout, oddsmpleout, index):
	if ind == 0:
		Z[0].append([X[0],Y[0]])
		Z[1].append([X[1],Y[1]])
		Z[2].append([X[2],Y[2]])
		Z[3].append([X[3],Y[3]])
		Z[4].append([X[4],Y[4]])
		
	else:
		O[0].append([X[0],Y[0]])
		O[1].append([X[1],Y[1]])
		O[2].append([X[2],Y[2]])
		O[3].append([X[3],Y[3]])
		O[4].append([X[4],Y[4]])


#change to arrays for slicing
npZ = numpy.array(Z)
npO = numpy.array(O)

#create a graph for each column to see if theres any oddities between columns
for i in range(5):
	columns = i
	title = "(x{},x{}) coloured zeros vs ones".format((i*2),((i*2)+1))

	One = (npO[columns,:,0],npO[columns,:,1])
	Zero = (npZ[columns,:,0],npZ[columns,:,1])



	data = (Zero, One)
	colors = ("blue", "orange")
	groups = ("0", "1")
	 #remark
	 #its definitely the '1' rows that result in an X and O pattern

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

	for data, color, group in zip(data, colors, groups):
		x, y = data
		ax.scatter(x, y, alpha=0.5, c=color, edgecolors='none', s=30, label=group)

	plt.title(title)
	plt.legend(loc=2)
	plt.show()
