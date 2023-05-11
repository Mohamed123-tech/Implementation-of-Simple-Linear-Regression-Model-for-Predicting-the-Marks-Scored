# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph. 

5. Predict the regression for marks by using the representation of the graph. 

6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Mohamed musharuf A

RegisterNumber: 212220040089

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')

df.head()

df.tail()

x = df.iloc[:,:-1].values

x

y = df.iloc[:,1].values

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

y_pred

y_test

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')

plt.plot(x_train,regressor.predict(x_train),color='purple')

plt.title("Hours vs Scores(Training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

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


## Output:

df.head()

![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/efe0a9b1-5170-48ce-8a0f-1268d72351aa)<br>
df.tail()

![229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/fc9c233f-4cdd-4ac0-97a9-765e6c3111cc)<br>
Array value of X<br>
![229978918-707c006d-0a30-4833-bf77-edd37e8849bb](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/c9216553-e876-45e1-8f53-a16f1695e196)

Array value of Y<br>
![229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/f7d9e7f5-9a5f-4e86-b3bb-c608b998a89c)

Values of Y prediction

![229979053-f32194cb-7ed4-4326-8a39-fe8186079b63](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/0d9e85b7-e1f1-40f7-9fc0-51ef151a0793)

Array values of Y test

![229979114-3667c4b7-7610-4175-9532-5538b83957ac](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/053e18aa-8b8c-42fa-8ee8-3741c93d4caa)

Training set graph

![229979169-ad4db5b6-e238-4d80-ae5b-405638820d35](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/437448bf-54b5-48b7-afbb-5490b2c9aa62)

Test set graph

![229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/fe77d06f-a7dc-47d4-a28f-e897aa865376)

Values of MSE, MAE and RMSE

![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/Mohamed123-tech/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/84170699/1b4709d8-055e-4005-888b-124184d19e95)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
