import numpy as np
import pandas as pd
df=pd.read_csv("C:/Users/User/Downloads/advertising.csv")
print(df.head)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
print(df.isnull().sum())
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,10))
sns.heatmap(df.corr())
plt.show()
x=np.array(df.drop(["Sales"],axis=1))
y=np.array(df["Sales"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
data=pd.DataFrame(data={"predicted_sales":ypred.flatten()})
print(data)
