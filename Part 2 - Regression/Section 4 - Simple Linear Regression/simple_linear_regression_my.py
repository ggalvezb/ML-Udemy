import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#read data
dataset=pd.read_csv("C:\\Users\\ggalv\\OneDrive\ML-Udemy\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splating the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X=StandardScaler()
# X_train=sc_X.fit_transform(X_train)
# X_test=sc_X.transform(X_test)

#Fittin Simple Linear Regression to the TRaining Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred=regressor.predict(X_test)

#Validing th training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (training set)')
plt.show()

#Validing th test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (training set)')
plt.show()