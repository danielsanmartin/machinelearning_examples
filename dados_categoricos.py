# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 08:38:35 2018

@author: Daniel
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./data/Data.csv')

X = dataset.iloc[:,:-1].values # todas as linhas menos a ultima colula
y = dataset.iloc[:,3].values # todas as linhas e a terceira coluna

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN',strategy='mean',axis=0) #axis -> todas as linhas da coluna 0
imp = imp.fit(X[:,1:3])

X[:,1:3] = imp.transform(X[:,1:3])

# Codificação dos Dados Categóricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelec = LabelEncoder()
onehot = OneHotEncoder(categorical_features = [0])

X[:,0] = labelec.fit_transform(X[:,0])

X = onehot.fit_transform(X).toarray()

# Feature scaling
# a ideia é deixar todas as variáveis na mesma escala.

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)