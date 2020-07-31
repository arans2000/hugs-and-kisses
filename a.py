#first group is top left skewed diagonally, second group is bottom right
#third group is  top left skewed horizontally, fourth group is a long diagonal bottom left to top right
#fifth group is a very dispersed grouping to the right

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
#take a row of this dataframe and convert it to numpy array
smplesize = 1000
nptrainarrsmpl = trainarr.iloc[:smplesize,:10].to_numpy()
#multiply that by the invers of 'a'
smpleout = nptrainarrsmpl.dot(ainverse)
#split it into odd and even elements to compare x0 v x1, x2 v x3, x4 v x5, etc.
evensmpleout=smpleout[:,1::2]
oddsmpleout=smpleout[:,0::2]


x0_vs_x1 = (evensmpleout[:,0],oddsmpleout[:,0])
x2_vs_x3 = (evensmpleout[:,1],oddsmpleout[:,1])
x4_vs_x5 = (evensmpleout[:,2],oddsmpleout[:,2])
x6_vs_x7 = (evensmpleout[:,3],oddsmpleout[:,3])
x8_vs_x9 = (evensmpleout[:,4],oddsmpleout[:,4])
data = (x0_vs_x1, x2_vs_x3, x4_vs_x5, x6_vs_x7, x8_vs_x9)
colors = ("white", "black", "white", "white", "white")#"orange", "yellow", "blue", "green")
groups = ("x0_vs_x1", "x2_vs_x3", "x4_vs_x5", "x6_vs_x7", "x8_vs_x9")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

for data, color, group in zip(data, colors, groups):
	x, y = data
	ax.scatter(x, y, alpha=0.5, c=color, edgecolors='none', s=30, label=group)

plt.title('x0 vs x1, x2 vs x3, etc.')
plt.legend(loc=0)
plt.show()
