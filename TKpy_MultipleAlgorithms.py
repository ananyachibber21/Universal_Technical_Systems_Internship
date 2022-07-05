# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot
# Load the dataset
dataset = pd.read_csv(r"C:\Users\DELL\Documents\Jupyter\Machine Learning Projects\Breast Cancer Detection\data.csv")
# Mapping string values to a number
dataset['diagnosis'] = dataset['diagnosis'].map({'B':0,'M':1}).astype(int)
# Segregating the dataset into independent variable x and dependent variable y
x = dataset.iloc[:,2:].values
y = dataset.iloc[:,1].values
# Split the data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Validating various ML algorithms by its accuracy - Model Score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Algorithm Comparison
results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
print(res)

class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
            
                                'TKPY_Multi_LR': [['float'], ['float'], ['Logistic Regression']],
                                'TKPY_Multi_LDA': [['float'], ['float'], ['Linear Discrimant Ananlysys']],
								                'TKPY_Multi_KNN': [['float'], ['float'], ['K Nearest Neighbors']],
								                'TKPY_Multi_CART': [['float'], ['float'], ['Decision Tree Classifier']],
                                'TKPY_Multi_NB': [['float'], ['float'], ['Gaussian']],
                                'TKPY_Multi_SVM': [['float'], ['float'], ['Support Vector Machine']]
							}
	def get_func_dict(self):
		return self.func_dict

def TKPY_Multi_LR(dummy):
	a = res[0]
	return float(a)

def TKPY_Multi_LDA(dummy):
	a = res[1]
	return float(a)

def TKPY_Multi_KNN(dummy):
	a = res[2]
	return float(a)

def TKPY_Multi_CART(dummy):
	a = res[3]
	return float(a)

def TKPY_Multi_NB(dummy):
	a = res[4]
	return float(a)

def TKPY_Multi_SVM(dummy):
	a = res[5]
	return float(a)
