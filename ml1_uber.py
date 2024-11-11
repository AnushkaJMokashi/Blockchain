import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('uber.csv')
df.head()
df.drop(['Unnamed: 0', 'key'], axis=1, inplace=True)
df.head()
df.info()
df.passenger_count.value_counts()
df = df[~ (df.passenger_count == 0) & ~(df.passenger_count == 208)]
df.shape
df.isnull().sum()
## Handle Range of latitude and longitude
df = df[
    (df.pickup_longitude < 180) & (df.pickup_longitude > -180) &
    (df.dropoff_longitude < 180) & (df.dropoff_longitude > -180) &
    (df.pickup_latitude < 90) & (df.pickup_latitude > -90) &
    (df.dropoff_latitude < 90) & (df.dropoff_latitude > -90)
]
df.shape
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['hour'] = df['pickup_datetime'].dt.hour
df.head()
df.drop(['pickup_datetime'], axis=1, inplace=True)
df.head()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
## Haversine distance

def haversine_distance(lon_1, lon_2, lat_1, lat_2):
    
    lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1, lat_2])  #Degrees to Radians
    
    
    diff_lon = lon_2 - lon_1
    diff_lat = lat_2 - lat_1
    

    km = 2 * 6371 * np.arcsin(
        np.sqrt(np.sin(diff_lat/2.0)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(diff_lon/2.0)**2)
    )
    
    return km
df['haversine_distance'] = haversine_distance(df['pickup_longitude'],df['dropoff_longitude'], df['pickup_latitude'],df['dropoff_latitude'])
df.head()
## Handling Outliers
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']

for col in numerical_cols:
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']

def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the DataFrame to remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_clean = remove_outliers(df, numerical_cols)
df_clean.shape
for col in numerical_cols:
    sns.scatterplot(x=df[col], y=df['fare_amount'])
    plt.title(f'Boxplot of {col}')
    plt.show()
sns.scatterplot(x=df['haversine_distance'], y=df['fare_amount'])
plt.xlim([0, 100])
df = df[~(df['haversine_distance'] == 0)]
sns.heatmap(df.corr(),annot=True)
Y = df.iloc[:,:1]
X = df.iloc[:,1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
## Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import r2_score

print('R2 score : ',r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

