import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from selenium import webdriver
from selenium.webdriver.common.by import By
import sched, time, datetime
import tkinter as tk
import threading
import datetime

data = pd.read_csv("GecmisVeriler.csv")
print(data.head())

data = data.drop(["Şimdi","Düşük","Fark %"],axis=1)
data['Tarih'] = pd.to_numeric(pd.to_datetime(data['Tarih'],format='%d.%m.%Y'))
y = data.drop(["Tarih","Açılış","Hac."],axis=1)
x = data.drop(["Yüksek"],axis=1)

x["Hac."] = x["Hac."].str.replace('K','000')
x["Hac."] = x["Hac."].str.replace('M','000000')
x["Hac."] = x["Hac."].str.replace('B','000000000')
x["Hac."] = x["Hac."].str.replace('.','')
x["Açılış"] = x["Açılış"].str.replace(".","")
x["Açılış"] = x["Açılış"].str.replace(",",".")
x["Hac."] = x["Hac."].str.replace(',','.')
y["Yüksek"] = y["Yüksek"].str.replace(".","")
y["Yüksek"] = y["Yüksek"].str.replace(",",".")

X = x.values
Y = y.values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
lin_reg_predict = lin_reg.predict(x_test)

from sklearn.preprocessing import PolynomialFeatures
x_poly = PolynomialFeatures(degree=2)
poly_reg = x_poly.fit_transform(x_train)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(poly_reg,y_train)
lin_reg_poly_predcit = lin_reg_poly.predict(x_poly.fit_transform(x_test))


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x_train,y_train)
rf_predcit = rf.predict(x_test)

from sklearn.metrics import r2_score
print("r2_score--------------------")
print("linear regrassion")
print(r2_score(y_test,lin_reg_predict))
print("polynımal regrassion")
print(r2_score(y_test,lin_reg_poly_predcit))
print("decision tree")
print(r2_score(y_test,dt_predict))
print( "random forest regrassıon ")
print(r2_score(y_test,rf_predcit))

date_str = "14.02.2023"
date = pd.to_datetime(date_str, format="%d.%m.%Y")
numeric_date = int(pd.Timestamp(date).value // 10**0)
print(numeric_date)
print("linear---------------")
print(lin_reg.predict([[numeric_date,1505.88,441.75000]]))
print("poly--------------------")
print(lin_reg_poly.predict(x_poly.fit_transform([[numeric_date,1505.88,441.75000]])))
print("Decision---------------")
print(dt.predict([[numeric_date,1505.88,441.75000]]))
print("RandomForest---------------")
print(rf.predict([[numeric_date,1505.88,441.75000]]))







