import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#read data
dataset=pd.read_csv("C:\\Users\\ggalv\\OneDrive\ML-Udemy\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#splating the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X=StandardScaler()
# X_train=sc_X.fit_transform(X_train)
# X_test=sc_X.transform(X_test)

