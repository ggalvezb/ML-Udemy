import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#read data
dataset=pd.read_csv("C:\\Users\\ggalv\\OneDrive\\ML-Udemy\\Part 1 - Data Preprocessing\\Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Taking care of missing data
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical data
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#splating the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)