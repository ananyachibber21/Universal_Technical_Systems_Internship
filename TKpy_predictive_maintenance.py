# %%
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_style('darkgrid')

# %%
dataset_name= r"D:\Users\aaa\OneDrive - The Northcap university\Desktop\predictive_maintenance.csv"

data= pd.read_csv(dataset_name)


# %%
data= data.drop(['UDI', 'Product ID'], axis=1)


# %%
from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()

label_encoder.fit(data['Type'])
data['Type']= label_encoder.transform(data['Type'])

label_encoder.fit(data['Target'])
data['Target']= label_encoder.transform(data['Target'])
label_encoder.fit(data['Failure_Type'])
data['Failure_Type']= label_encoder.transform(data['Failure_Type'])

# %%
data

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(data.drop(['Failure_Type', 'Target'], axis=1),
data['Failure_Type'], test_size= 0.3, random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)

# %%
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

classifier= []
imported_as= []


# %%
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
classifier.append('Logistic Regression')
imported_as.append('lr')


# %%
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
classifier.append(' K Nearest Neighbors')
imported_as.append('knn')

# %%
from sklearn.svm import SVC
svc= SVC()
classifier.append('Support Vector Machine')
imported_as.append('svc')

# %%
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()
classifier.append('Random Forest')
imported_as.append('rfc')

# %%
from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
classifier.append('Naive Bayes')
imported_as.append('nb')

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
dt= DecisionTreeClassifier()

classifier.append('Decision Tree')
imported_as.append('dt')

# %%
classifiers= pd.DataFrame({'Classifier':classifier, 'Imported as': imported_as})



# %%
import plotly.express as px
class Modelling:
    def __init__(self, x_train, y_train, x_test, y_test, models):
        self.x_train= x_train
        self.x_test= x_test
        self.y_train= y_train
        self.y_test= y_test
        self.models= models
    
    def fit(self):
        model_acc= []
        model_time= []

        for i in self.models:
            start= time.time()
            if i == 'knn':
                accuracy= []
                for j in range(1, 200):
                    kn= KNeighborsClassifier(n_neighbors= j)
                    kn.fit(self.x_train, self.y_train)
                    predK= kn.predict(self.x_test)
                    accuracy.append([accuracy_score(self.y_test, predK), j])
                temp= accuracy[0]
                for m in accuracy:
                    if temp[0] < m[0]:
                        temp= m
                i= KNeighborsClassifier(n_neighbors=temp[1])
            i.fit(self.x_train, self.y_train)
            model_acc.append(accuracy_score(self.y_test, i.predict(self.x_test)))
            stop= time.time()
            model_time.append((stop-start))
            #print(i, 'has been fit')
        self.models_output= pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})
    
    def results(self):
        models= self.models_output
        models= models.sort_values(by= ['Accuracy', 'Runtime (s)'], ascending=[False, True]).reset_index().drop('index', axis=1)
        self.best= models['Models'][0]
        models['Models']= models['Models'].astype(str).str.split("(", n=2, expand= True)[0]
        models['Accuracy']= models['Accuracy'].round(5)*100
        self.models_output_cleaned= models
        return (models)
    
    def best_model(self, type):
        if type== 'model':
            return(self.best)
        elif type== 'name':
            return(self.models_output_cleaned['Models'][0])
    
    def best_model_accuracy(self):
        return(self.models_output_cleaned['Accuracy'][0])
    
    def best_model_runtime(self):
        return(round(self.models_output_cleaned['Runtime (s)'][0], 3))

    def best_model_predict(self, x_test):
        x_test= scaler.transform(x_test)
        return(self.best.predict(x_test))

    def best_model_confusion_matrix(self):
        return(confusion_matrix(self.y_test, self.best.predict(self.x_test)))

    def best_model_clmatrix(self):
        return(classification_report(self.y_test, self.best.predict(self.x_test)))
    def model_3d_plot(self):
        fig= px.scatter_3d(data, x= 'Air temperature [K]', y='Rotational speed [rpm]', z='Torque [Nm]', color='Failure_Type')
        return(fig.show())

# %%
models_to_test= [rfc, lr, knn, svc, nb, dt]

# %%
classification= Modelling(x_train, y_train, x_test, y_test, models_to_test)
classification.fit()
classification.results()


# %%
file= 'best_model.sav'
import pickle
pickle.dump(classification, open(file, 'wb'))

# %%



