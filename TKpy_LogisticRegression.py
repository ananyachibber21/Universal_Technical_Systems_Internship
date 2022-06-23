# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics

# Load the dataset
dataset = pd.read_csv(r"C:\Users\DELL\Documents\Jupyter\Machine Learning Projects\Brand Sale Prediction\sale_data.csv")

# Segregating the dataset into independent variable x and dependent variable y
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split the data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

# Feature Scaling - we scale our data to make all the featured contribute equally to the result
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Train the algorithm
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# Prediction for all Test Data
y_pred = model.predict(x_test)

class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
            
                                'TKPY_logistic_pred': [['array'], ['float'], ['Predicted values']],
                                'TKPY_logistic_Confusion': [['int'], ['matrix'], ['Confusion Matrix']],
			        'TKPY_logistic_r2_score': [['int'], ['float'], ['r^2 score of Model']],
			        'TKPY_logistic_coef': [['int'], ['array'], ['Coefficient values']],
                                'TKPY_logistic_intercept': [['int'], ['float'], ['Intercept values']]
							}
	def get_func_dict(self):
		return self.func_dict

def TKPY_logistic_pred(dummy):
	a = model.predict(sc.transform([dummy]))
	return float(a)

def TKPY_logistic_Confusion(dummy):
	a = confusion_matrix(y_test,y_pred)
	a = a + dummy - dummy
	return np.array(a)

def TKPY_logistic_r2_score(dummy):
	a = model.score(x_test,y_test)
	a = a + dummy - dummy
	return(a)

def TKPY_logistic_coef(dummy):
	a=model.coef_
	return a.flatten()

def TKPY_logistic_intercept(dummy):
	a=model.intercept_
	return a
