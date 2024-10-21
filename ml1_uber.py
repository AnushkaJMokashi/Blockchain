import pandas as pd
 import numpy as np
 import seaborn as sns
 import matplotlib.pyplot as plt
 %matplotlib inline
 df = pd.read_csv('uber.csv')
 df.head()

 df.dropna(inplace=True)
 df.isnull().sum()

fare_amt = [fare for fare in df.fare_amount if fare>=0]
len(fare_amt)
199982
neg_fare_amt = [fare for fare in df.fare_amount if fare < 0]
len(neg_fare_amt)
17
df.fare_amount = [fare if fare >= 0 else 0 for fare in df.fare_amount]
fare_amt = [fare for fare in df.fare_amount if fare==0]
len(fare_amt)

long_lati_cols = 
['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
 for col in long_lati_cols:
    df[col] = [round(i,4) for i in df[col]]
 df


from math import *
 def distance_formula(longitude1,latitude1,longitude2,latitude2):
    travel_dist = []
    for pos in range (len(longitude1)):
        lon1, lan1, lon2, lan2 = map(radians, [longitude1[pos], 
latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_lon = lon2 - lon1
        dist_lan = lan2 - lan1
        a=sin(dist_lan/2)**2 + cos(lan1) * cos(lan2) * 
sin(dist_lon/2)**2
        c = 2 * asin(sqrt(a)) * 6371
        travel_dist.append(c)
    return travel_dist
        
 df['dist_travel_km'] = 
distance_formula(df.pickup_longitude.to_numpy(),df.pickup_latitude.to_numpy(), df.dropoff_longitude.to_numpy(), df.dropoff_latitude.to_numpy())
df['dist_travel_km']


 from datetime import datetime
 df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
 df

 df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%m/%d/%Y, %H:%M:%S")
 df.pickup_datetime

 df.drop("pickup_datetime",axis=1,inplace=True)
 df

sns.boxplot(x=df['fare_amount'])


df.plot(kind = "box", subplots = True, layout = (6,2),figsize=(15,20))
 plt.show() 

def remove_outlier(df1, col) : 
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df1[col], lower_bound, upper_bound)
    return df1

def treat_outliers(df1, col_list):
    for c in col_list:
        df1 = remove_outlier(df1, c)
    return df1
df = treat_outliers(df, df.iloc[:,0::])
df.plot(kind = "box", subplots = True, layout = (6, 2), figsize = (15, 20))
plt.show()
corr = df.corr()
fig, axis = plt.subplots(figsize = (10, 6))
sns.heatmap(df.corr(), annot = True)

from sklearn.model_selection import train_test_split

df_x = df[['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'dist_travel_km', 'hour', 'day', 'month', 'year', 'dayofweek']]

df_y = df['fare_amount']

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 1)


from sklearn.linear_model import LinearRegression

model.fit(x_train, y_train)

y_pred_lin = model.predict(x_test)
print(y_pred_lin)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor (n_estimators= 100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
y_pred_rf


cols = ['Model', 'RMSE', 'R-Squared']
result_tabulation = pd.DataFrame(columns = cols)

from sklearn.metrics import r2_score, mean_squared_error

reg_RMSE = np.sqrt(mean_squared_error(y_test, y_pred_lin))
reg_squared = r2_score(y_test, y_pred_lin)

full_metrics = pd.Series({'Model' : "Linear Regression", 'RMSE' : reg_RMSE, 'R-Squared' : reg_squared})

result_tabulation = result_tabulation._append(full_metrics, ignore_index = True)

result_tabulation

rf_RMSE = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_squared = r2_score(y_test, y_pred_rf)

full_metrics = pd.Series({'Model' : "Random Forest", 'RMSE' : rf_RMSE, 'R-Squared' : rf_squared})
result_tabulation = result_tabulation._append(full_metrics, ignore_index = True)

result_tabulation
 
