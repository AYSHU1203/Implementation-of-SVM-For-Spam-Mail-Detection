# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.
 
2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer.

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

8.Find the accuracy of the model.

9.End the Program.
 

## Program:
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: AYSHWARIYA J
RegisterNumber:212224230030  
*/
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![Image-1](https://github.com/user-attachments/assets/d83ce202-d124-4afd-94ff-703c5e5fc7e9)

![Image-2](https://github.com/user-attachments/assets/0bd2a88e-cd09-4873-b0c0-749cfc9ac5f9)

![Image-3](https://github.com/user-attachments/assets/39f527ce-541d-4c8c-8019-fb65cad62f20)

![Image-4](https://github.com/user-attachments/assets/8aafe725-8dc0-4803-88c1-a823bfcf5ee9)

![Image-5](https://github.com/user-attachments/assets/8ff7db68-1e63-4994-a01c-83c35ee2d4d3)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
