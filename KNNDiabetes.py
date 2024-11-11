import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
df=pd.read_csv("diabetes.csv") #Reading the Dataset
df.head()
df.dtypes
df["Glucose"].replace(0,df["Glucose"].mean(), inplace=True)
df["BloodPressure"].replace(0,df["BloodPressure"].mean(), inplace=True)
df["SkinThickness"].replace(0,df["SkinThickness"].mean(), inplace=True)
df["Insulin"].replace(0,df["Insulin"].mean(), inplace=True)
df["BMI"].replace(0,df["BMI"].mean(), inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

df_scaled = pd.DataFrame(ss.fit_transform(df), columns=ss.get_feature_names_out())
df_scaled.head()

X = df.iloc[:, :8]
Y = df.iloc[:, 8:]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.46,random_state=46)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
operations = [('knn', knn)]
pipe = Pipeline(operations)

# Set up GridSearchCV for hyperparameter tuning
k_values = list(range(1, 20))
param_grid = {'knn__n_neighbors': k_values}
full_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

full_classifier.fit(X_train, y_train)

best_params = full_classifier.best_estimator_.get_params()
print(best_params)


y_pred = full_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Error Rate: {error_rate}')
cm = confusion_matrix(y_test, y_pred)
print(cm)

CMD = ConfusionMatrixDisplay(cm).plot()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Precision: {precision}')
print(f'Recall: {recall}')

print(classification_report(y_test, y_pred))
def apply_model(model):#Model to print the scores of various models
    model.fit(X_train,Y_train)
    print("Training score = ",model.score(X_train,Y_train))
    print("Testing score = ",model.score(X_test,Y_test))
    print("Accuracy = ",model.score(X_test,Y_test))
    Y_pred = model.predict(X_test)
    print("Predicted values:\n",Y_pred)
    print("Confusion Matrix:\n",confusion_matrix(Y_test,Y_pred))
    print("Classification Report:\n",classification_report(Y_test,Y_pred))
knn = KNeighborsClassifier(n_neighbors=5) #KNN Model
apply_model(knn)

