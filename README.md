# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
- The data is split into training and testing sets after preprocessing.
- A Decision Tree Classifier (entropy criterion) is trained on the training data.
- The model predicts employee attrition based on decision rules.
- Accuracy is calculated by comparing predictions with actual test results.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mahith M
RegisterNumber:  212225220061
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy",random_state=100)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
sample=[[0.5,0.8,9,260,6,0,1,2]]
print("Prediction for sample:",dt.predict(sample))
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=x.columns,class_names=["stayed","left"],filled=True,rounded=True,fontsize=10)
plt.show()
```

## Output:
<img width="808" height="590" alt="image" src="https://github.com/user-attachments/assets/63687daf-107c-4467-a0ef-ea42b17dc656" />
<img width="943" height="643" alt="image" src="https://github.com/user-attachments/assets/daaeec93-de31-4d5d-8d0c-e04dd3ca3fcc" />
<img width="1106" height="645" alt="image" src="https://github.com/user-attachments/assets/af999af1-cf2f-488e-a844-9079a16a432e" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
