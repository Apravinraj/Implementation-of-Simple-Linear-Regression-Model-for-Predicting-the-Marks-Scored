## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212222240079
RegisterNumber:  PRAVIN RAJ A
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
Output:
Input Data

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/9ba4b396-59e8-4c58-8f10-14fbfa6190ce)


Head and Tail of the Data set

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/9b4a59a5-896d-4b32-a3e9-687adabab2d6)

X Values and Y Values

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/c92eb60c-9ca5-4228-92df-84a1b41eed14)


Prediction Values

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/b367535e-7768-479b-90db-7a636102c323)


Training Set

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/94ab925a-a148-4dd5-9fe9-7437784bd343)

Test Set

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/7bbc7da3-58ac-4410-a55d-80f2c1caf1da)


Values of MSE , MAE and RMSE

![image](https://github.com/Apravinraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707879/ba81a21a-d444-4e7d-94ef-aa15f5ea7e3b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
