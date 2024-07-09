import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy import stats

iris = load_iris()                 #df = pd.read_csv('')
df = pd.DataFrame(data = iris.data,columns=iris.feature_names)
df['target'] = iris.target

print(df.info())                    #數據基本訊息
print(df.isnull().sum())            #檢查缺失值
df.fillna(df.mean(),inplace=True)   #處理缺失值

print(df.duplicated().sum())        #檢查重複數據
df.drop_duplicates(inplace=True)    #去除重複數值
print(df.shape[0])                  #行數

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))    #選擇數值類型numeric
df = df[(z_scores<3).all(axis=1)]     #去除離散值

#df = pd.get_dummies(df,columns=['target'])  目標類別數值型

scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(X=df[df.columns[:-1]]) #數據標準化

print(df.head())