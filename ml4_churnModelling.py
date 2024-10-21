import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
df = pd.read_csv("Churn_Modelling.csv")
df.describe()
df.head()
x = df.drop(["Surname","Geography","RowNumber","Gender","Exited"],axis = 1)
y = df["Exited"]
sns.countplot(x=y)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
x_scaled
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state = 2,test_size = 0.25)
x_train.shape
x_test.shape
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(100,100,100),random_state=2,max_iter=100,activation = 'relu')
ann.fit(x_train,y_train)
y_pred = ann.predict(x_test)
y_pred
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,classification_report
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
