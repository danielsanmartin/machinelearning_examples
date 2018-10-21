#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:01:28 2018

@author: daniel
"""


import pandas as pd


dataset = pd.read_csv('./data/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

# codificando variaveis categoricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()

X[:,1] = labelencoder_1.fit_transform(X[:,1])

labelencoder_2 = LabelEncoder()
X[:,2] = labelencoder_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

# eliminando a varialve Dummy
X = X[:,1:]

# dividindo o treinamento entre treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0)

# normalizando os dados (Feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_treino = sc.fit_transform(X_train)
X_teste = sc.fit_transform(X_test)

# montando a RNA
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu', input_dim=11))
model.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, epochs=100)

y_predit = model.predict(X_test)

# transformo em 0 e 1 para para binarizar o resultado
y_predit = (y_predit > 0.5) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predit)

acuracy = (cm[0,0]+cm[1,1])/cm.sum()