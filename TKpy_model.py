import os
import numpy as np
from pred_mt import Modelling
import pickle
file= 'best_model.sav'
models= pickle.load(open(file, 'rb'))


class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	'TKPY_model': [['int'], ['string'], ['Name of the Machine Learning Model used']],
								'TKPY_confusion_matrix': [['int'], ['matrix'], ['Confusion Matrix of the Model']],
								'TKPY_accuracy': [['int'], ['float'], ['Accuracy Score of the Model']],
                                'TKPY_runtime' : [['float'], ['float'], ['Runtime of the Model in seconds']],
                                'TKPY_intercept': [['int'], ['float'], ['Intercept of the Model']],
                                'TKPY_coef': [['int'], ['array'], ['Coefficients of the Model']],
								'TKPY_predict' : [['array'], ['int'], ['Predicted Value for a given set of input']],
                                'TKPY_3d_plot' : [['inplot'], ['int'], ['3D Plot For The Failure Types in the Data']]
							}
	def get_func_dict(self):
		return self.func_dict

def TKPY_model(dummy):
    return models.best_model(type='name')

def TKPY_confusion_matrix(dummy):
    return models.best_model_confusion_matrix()

def TKPY_accuracy(dummy):
    return models.best_model_accuracy()

def TKPY_runtime(dummy):
    return models.best_model_runtime()

def TKPY_predict(list):
    arr= np.array([list])
    list= list.reshape(1, -1)
    return models.best_model_predict(list)

def TKPY_3d_plot(dummy):
    os.system('start /min cmd /high /k "cd TKExtLib\Python &  python -c "from model import *; models.model_3d_plot() " &exit"')
    return dummy

# print(TKPY_model(1))
# print(TKPY_confusion_matrix(1))
# print(TKPY_accuracy(1))
# print(TKPY_runtime(1))
# a= np.array( [1,298.9,309.1,1383,54.9,145])
# a= a.reshape(1, -1)
# print(TKPY_predict(a))
# print(TKPY_3d_plot(1))

