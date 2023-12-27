import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('C:/Users/User/Downloads/dataset.csv')
df.head()
df.shape
df.describe()
df_missing_values=df.isnull().sum()
print(df_missing_values)
classes=df['Class'].value_counts()
print(classes)
sns.countplot(x='Class',data=df)
plt.title('no.of fraud vs non fraud transactions')
plt.show()
data_fraud=df[df['Class']==1]
data_non_fraud=df[df['Class']==0]
plt.figure(figsize=(8,5))
ax=sns.distplot(data_fraud['Time'],label='fraudt',hist=False)
ax=sns.distplot(data_non_fraud['Time'],label='non fraud',hist=False)
ax.set(xlabel='seconds elapsed between the transaction and the first transaction')
plt.show()
df.drop('Time',axis=1,inplace=True)
plt.figure(figsize=(8,5))
ax=sns.distplot(data_fraud['Amount'],label='fraudulent',hist=False)
ax=sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
ax.set(xlabel='Transaction Amount')
plt.show()
from sklearn.model_selection import train_test_split
X=df.drop(['Class'],axis=1)
Y=df['Class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=100)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train['Amount']=scaler.fit_transform(X_train[['Amount']])
print(X_train.head())
X_test['Amount']=scaler.transform(X_test[['Amount']])
print(X_test.head())
from sklearn import metrics
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,f1_score
results=pd.DataFrame(columns=['Model Name','Accuracy','f1_score','ROC'])
def draw_roc(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr,label='ROC curve(area=%0.2f)'%auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate or [1-True Negetive Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return None
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(C=0.01)
logistic_model=logistic.fit(X_train,Y_train)
def display_test_results(model_name,model):
  Y_test_pred=model.predict(X_test)
  c_matrix=metrics.confusion_matrix(Y_test,Y_test_pred)
  print(c_matrix)
  cm_display=ConfusionMatrixDisplay(confusion_matrix=c_matrix)
  cm_display.plot(cmap=plt.cm.Blues)
  plt.show()
  print(classification_report(Y_test,Y_test_pred))
  TP=c_matrix[1,1]
  TN=c_matrix[0,0]
  FP=c_matrix[0,1]
  FN=c_matrix[1,0]
  print("accuracy:",metrics.accuracy_score(Y_test,Y_test_pred))
  print("sencivity:",TN/float(TP+FP))
  print("f1-score:",f1_score(Y_test,Y_test_pred))
  Y_test_pred_proba=model.predict_proba(X_test)[:,1]
  roc_auc=metrics.roc_auc_score(Y_test,Y_test_pred_proba)
  draw_roc(Y_test,Y_test_pred_proba)
  results.loc[len(results)]=[model_name,metrics.accuracy_score(Y_test,Y_test_pred),f1_score(Y_test,Y_test_pred),roc_auc]
  return None
display_test_results("LogisticRegression",logistic_model)
from sklearn.tree import DecisionTreeClassifier
decision_tree_model=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,min_samples_leaf=100,min_samples_split=100)
decision_tree_model.fit(X_train,Y_train)
DecisionTreeClassifier(max_depth=5,min_samples_leaf=100,min_samples_split=100,random_state=100)
display_test_results("Decision Tree",decision_tree_model)
print(results.sort_values(by="ROC",ascending=False))
