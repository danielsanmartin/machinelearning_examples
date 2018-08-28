#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:23:22 2018

@author: daniel
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('./data/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor

# n_estimators = 5 significa que serão criadas 5 árvores
preditor = RandomForestRegressor(n_estimators = 5, random_state=0)

preditor.fit(X, y)

print('O salário previsto para 6.8 é {}'.format(preditor.predict(6.8)))

# Rearrajo os dados para facilitar a visualização serrilhada do decision tree
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)

# Visualização dos dados
plt.scatter(X,y,color='red')
plt.plot(X_grid, preditor.predict(X_grid), color='blue')
plt.title('Regressão Polinomial')
plt.xlabel('Posição')
plt.ylabel('Salário')
plt.show()