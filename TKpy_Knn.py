# Importing the libraries
import numpy as np
import pandas as pd
import math
from math import sqrt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error

dataset = pd.read_csv(r"C:\\Users\\DELL\\Documents\\Jupyter\\Machine Learning Projects\\Smart Phone Sale Prediction\\smartphone.csv")

# Mapping Gender Data to Binary value
gender_set = set(dataset['Gender'])
dataset['Gender'] = dataset['Gender'].map({'Male': 1,'Female': 0}).astype(int)

# print(dataset.head())

# Segregating the dataset into independent variable x and dependent variable y
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split the data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Feature Scaling - we scale our data to make all the featured contribute equally to the result
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Finding the best K value
error = []
from sklearn.neighbors import KNeighborsClassifier

# Calculating error for K values between 1 and 40
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i, metric="minkowski",p=2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test) # Predicting the test data
    error.append(np.mean(y_pred!=y_test))
# print(error)

# To find the K value with the formula and without visualizing the graph
list_a = np.array(error)
list_b = []
l_a=[]
l_b=[]
for i in range(1,40):
    list_b.append(i)
l_a.append(min(list_a))
l_b.append(min(list_b))
def run_euc(l_a,l_b):
  return np.array([[np.linalg.norm(i-j) for j in l_b] for i in l_a])
z = run_euc(l_a, l_b)
t = np.round(z)

# Train the algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1, metric="minkowski",p=2) # Calculating for K = 6
# For Euclidean distance we have to specify metric as minkowski and p=2
# For Manhattan distance we have to specify metric as minkowski and p=1
model.fit(x_train, y_train)

# Prediction for all Test Data
y_pred = model.predict(x_test)

class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
            
                                'TKPY_knn_pred': [['int','int','int'], ['int'], ['Predicted Value']],
                                'TKPY_knn_r2_score': [['float'], ['float'], ['Accuracy Value']],
                                'TKPY_knn_precision':[['int'], ['float'], ['Precision Value']],
				'TKPY_knn_rmse': [['int'], ['float'], ['Root Mean Square Error score of Model']],
                                'TKPY_knn_mae': [['int'], ['float'], ['Mean Absolute Error of Model']],
                                'TKPY_knn_recall': [['int'], ['float'], ['Recall Score of Model']],
                                'TKPY_knn_Confusion': [['int'], ['matrix'], ['Confusion Matrix']]
                                
		}
	def get_func_dict(self):
		return self.func_dict

# Prediction value
def TKPY_knn_pred(p, q , r):
    x_pred= np.array([p, q, r])
    return (model.predict(x_pred.reshape(1,3)))

def TKPY_knn_r2_score(dummy):
    a = model.score(x_test,y_test)
    a = a + dummy - dummy
    return a

def TKPY_knn_precision(dummy):
    a = metrics.precision_score(y_test, y_pred)
    a = a + dummy - dummy
    return a

def TKPY_knn_rmse(dummy):
    a = sqrt(mean_squared_error(y_test, y_pred))
    a = a + dummy - dummy
    return a

def TKPY_knn_mae(dummy):
    a = sqrt(mean_absolute_error(y_test, y_pred))
    a = a + dummy - dummy
    return a

def TKPY_knn_recall(dummy):
    a = metrics.recall_score(y_test, y_pred)
    a = a + dummy - dummy
    return a

def TKPY_knn_Confusion(dummy):
    a = confusion_matrix(y_test,y_pred)
    a = a + dummy - dummy
    return np.array(a)
