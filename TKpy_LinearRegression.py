import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
dataset = pd.read_csv(r"C:\Users\DELL\Desktop\Machine-Learning-with-Python\House Price Prediction\house_price.csv")

# Segregate Dataset into input x and output y
x = dataset[['area']]
y = dataset[['price']]

# Split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Train the algorithm
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
            
                                'TKPY_linear_mse': [['int'], ['float'], ['Mean Square Error score of Model']],
                                'TKPY_linear_mae': [['int'], ['float'], ['Mean Absolute Error score of Model']],
								'TKPY_linear_rmse': [['int'], ['float'], ['Root Mean Square Error score of Model']],
                                'TKPY_linear_coef': [['int'], ['float'], ['Coefficient values']],
                                'TKPY_linear_intercept': [['int'], ['float'], ['Intercept values']],
                                'TKPY_linear_r2_score': [['int'], ['float'], ['r^2 score of Model']],
                                'TKPY_linear_pred': [['int'], ['float'], ['Predicted values']]
							}
	def get_func_dict(self):
		return self.func_dict

def TKPY_linear_mse(dummy):
    # y_pred=pred(dummy)
    return (metrics.mean_squared_error(y_test,y_pred))

def TKPY_linear_mae(dummy):
    # y_pred=pred(dummy)
    return metrics.mean_absolute_error(y_test,y_pred)

def TKPY_linear_rmse(dummy):
    # y_pred=pred(dummy)
    return np.sqrt(metrics.mean_squared_error(y_test,y_pred))

def TKPY_linear_coef(dummy):
    return (model.coef_)

def TKPY_linear_intercept(dummy):
    return(model.intercept_)

def TKPY_linear_r2_score(dummy):
    return model.score(x_test,y_test)

def TKPY_linear_pred(dummy):
    return float(model.predict([[dummy]]))

# print(TKPY_linear_pred(1500))
# print(TKPY_linear_r2_score(1500))
# print(TKPY_linear_coef(1500))
# print(TKPY_linear_intercept(1500))
# print(TKPY_linear_mse(1500))
# print(TKPY_linear_mae(1500))
# print(TKPY_linear_rmse(1500))

# def main(z):
#     a = pred(z)
#     b = r2_score(z)
#     c = coef(z)
#     d = intercept(z)
#     e = mse(z)
#     f = mae(z)
#     g = rmse(z)
#     return np.array([a,b,c,d,e,f,g])

# print(main(20))