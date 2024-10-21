import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv('emails.csv')
df.head()

null_values = df.isnull().sum() > 0
null_values.sum()

x = df.iloc[:,1:3001]
y = df.iloc[:,-1:3002]
##Standard Scaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_scaled = pd.DataFrame(ss.fit_transform(x),columns=ss.get_feature_names_out())
x_scaled

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

KNeighborsClassifier()
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


 for i in range(15):
    knn2 = KNeighborsClassifier(n_neighbors=i+1)
    knn2.fit(X_train,y_train)
    y_pred2 = knn2.predict(X_test)
    print('Accuracy : {acc} , HyperParameter :  {n}'.format(acc=accuracy_score(y_pred2,y_test), n = i+1))


 import matplotlib.pyplot as plt
 from sklearn.inspection import DecisionBoundaryDisplay
 from sklearn.svm import SVC


svc = SVC(kernel="rbf",gamma=0.5, C=1.0)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_pred,y_test)

svc = SVC(kernel="linear", C=1.0)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_pred,y_test)

svc = SVC(kernel="sigmoid", C=1.0)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_pred,y_test)

kernels = ["linear","rbf","sigmoid","poly"]
for i in range(5):
 for j in kernels:
     svc2 = SVC(kernel=j, C=i+1)
     svc2.fit(X_train,y_train)
     y_pred = svc2.predict(X_test)
     print("Accuracy : {acc}, Kernel : {k}, C : {x}".format(acc=accuracy_score(y_pred,y_test),k=j, x = i+1))
