#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:32:51 2018

@author: daniel
"""


import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./data/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# Como o dataset é muito pequeno e de exemplo, não iremos dividir ele em dados
# de treinamento e de teste

from sklearn.preprocessing import PolynomialFeatures
poli = PolynomialFeatures(degree=4)

from sklearn.linear_model import LinearRegression
predictor = LinearRegression()

# Adequando as características do polinomio

X_poli = poli.fit_transform(X)
predictor.fit(X_poli, y)

# visualizar os resultados

plt.scatter(X,y,color='red')
plt.plot(X, predictor.predict(X_poli), color='blue')
plt.title('Regressão Polinomial')
plt.xlabel('Posição')
plt.ylabel('Salário')
plt.show()

