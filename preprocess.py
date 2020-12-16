# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:06:41 2020

@author: vinod
"""
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import make_union
from sklearn.pipeline import Pipeline,FeatureUnion
import pyodbc 
conn = pyodbc.connect('Driver={SQL Server};Server=DESKTOP-4L6HO2L\SQLEXPRESS;Database=reports;Trusted_Connection=yes;')# Creating Cursor   
cursor = conn.cursor()   
query="select * from dbo.practice"
df=pd.read_sql(query, conn)
df_cat=df.select_dtypes(exclude=['float64'])
df_float=df.select_dtypes(include=['float64'])
df_int=df.select_dtypes(include=['int'])
columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [0])], 
                                      remainder='passthrough') 
columnTransformer1 = ColumnTransformer([('sclar', 
                                        StandardScaler(), 
                                        [0])], 
                                      remainder='passthrough') 
a=columnTransformer1.fit_transform(df_float) 
b=columnTransformer.fit_transform(df_cat)

c=np.hstack((a,b))

try to check :
c[1]
array([1.95586034e+00, 1.51377590e+05, 4.43898530e+05, 1.91792060e+05,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00])
