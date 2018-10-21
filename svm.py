#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:54:32 2018

@author: daniel
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv('./data/Social_Network_Ads.csv')

dataset.drop(['User ID'], axis=1, inplace=True)

X = dataset.iloc[:,:-1].values # todas as linhas menos a última coluna
y = dataset.iloc[:,-1:].values


# Transforma a variável categorica em numérica
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelec = LabelEncoder()
onehot = OneHotEncoder(categorical_features = [0])

X[:,0] = labelec.fit_transform(X[:,0])

X = onehot.fit_transform(X).toarray()

X = X[:,1:]

    
# Feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.25, random_state = 0)


classificador = SVC(random_state = 0)
classificador.fit(X_treino, y_treino)

y_pred = classificador.predict(X_teste)

# gero um relatorio para avaliar a precisao do meu modelo
from sklearn.metrics import classification_report
print(classification_report(y_teste, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_teste, y_pred))