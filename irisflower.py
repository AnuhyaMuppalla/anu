import numpy as np
import pandas as pd
df=pd.read_csv("C:/Users/User/Downloads/IRIS.csv")
print(df.head)
print(df.describe())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
z=le.fit_transform(df["species"])
import matplotlib.pyplot as plt
plt.scatter(df["sepal_width"],df["sepal_length"],c=z)
plt.show()
x=df.drop("species",axis=1)
y=df["species"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
arr=np.array([[4,1.9,5,6.6]])
pred=knn.predict(arr)
print(pred)
