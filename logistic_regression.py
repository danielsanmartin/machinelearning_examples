#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:54:32 2018

@author: daniel
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('./data/Social_Network_Ads.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# tarefa: codificacao das caracteristicas do sexo (labelencoder)


X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.25, random_state = 0)
    
# Feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

classificador = LogisticRegression(random_state = 0)
classificador.fit(X_treino, y_treino)

y_pred = classificador.predict(X_teste)

from sklearn.metrics import confusion_matrix
cn = confusion_matrix


Este trabalho é do Logistic regression. Depois vou fazer o mesmo com o SVM.