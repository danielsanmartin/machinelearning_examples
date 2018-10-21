#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daniel
@date: 18/09/2018


Problema de Classificação

Descobrir o melhor algoritmo para classificar o seguinte conjunto de dados:
(Machine learning classifiers: logistic regression, random forest, decision trees and SVM )

Este conjunto de dados contém incidentes derivados do sistema de Relatórios de 
Incidentes Criminais da SFPD (San Francisco Police Departament). Os dados 
variam de 1/1/2003 a 5/13/2015. O conjunto de treino e o conjunto de testes 
rodam todas as semanas, ou seja, a semana 1,3,5,7 ... pertence ao grupo de 
teste, a semana 2,4,6,8 pertence ao conjunto de treino.

O objetivo é classificar o tipo de crime (Category).

Seguem as descrições das características do Dataset:

     Dates - data e hora do incidente do crime
     Category - categoria do incidente criminal (apenas em train.csv). 
     Descript - descrição detalhada do incidente do crime (somente em train.csv)
     DayOfWeek - o dia da semana
     PdDistrict - nome do Distrito do Departamento de Polícia
     Resolution - como o incidente do crime foi resolvido (somente em train.csv)
     Address - o endereço aproximado do incidente do crime
     X - Longitude
     Y - Latitude
 
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

pd.options.display.max_columns = None # mostra todas as colunas no console



def confusion_matrix_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    total = 0
    diagonal = 0
    for i in range(0, len(cm)):
        diagonal = diagonal + cm[i,i]
        for j in range(0, len(cm)):
            total = total + cm[i,j]
    
    return diagonal / total    

###############################################################################
# PRE-PROCESSAMENTO / PREPARANDO OS DADOS
###############################################################################

# DADOS DE TREINAMENTO
# Baixar em https://www.kaggle.com/c/sf-crime/data
train_data = pd.read_csv('./crime-train.csv')


del train_data['Descript'] # Removo pois nao esta nos dados de teste
del train_data['Resolution'] # Removo pois nao esta nos dados de teste
del train_data['Address'] # com o lat/long (X,Y) não preciso da rua

# convertendo as datas (strings) para ordinais para facilitar o processamento
train_data['Dates'] = pd.to_datetime(train_data["Dates"])
train_data = train_data[train_data['Dates'] > '2014-01-01 00:00:00']

# utilizo uma parcela dos dados para economizar memória.
train_data['Month'] = train_data['Dates'].apply(lambda x: x.month)
train_data['Day'] = train_data['Dates'].apply(lambda x: x.day)
train_data['Hour'] = train_data['Dates'].apply(lambda x: x.hour)

del train_data['Dates']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelec = LabelEncoder()

train_data['DayOfWeek'] = labelec.fit_transform(train_data['DayOfWeek'])
train_data['PdDistrict'] = labelec.fit_transform(train_data['PdDistrict'])

X = train_data.drop(['Category'],axis=1) # variaveis independentes
y = train_data['Category'] # variaves dependente

del train_data

# Transforma a variável categorica em numérica
onehot = OneHotEncoder(categorical_features = [0,1])
X = onehot.fit_transform(X).toarray()

# Feature scaling
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state = 0)


# TREINANDO O CLASSIFICADOR
from sklearn.metrics import accuracy_score
    
print('-- Decision Tree Classifier --------------------')
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

print('Accuracy score: ', accuracy_score(y_test, y_pred_dtc))
print('Confusion matrix accuracy: %f'% confusion_matrix_accuracy(y_test, y_pred_dtc))


# cross_val_score fatia o conjunto de treino em 10 e roda o algoritmo para cada um
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
print('Cross Validation Score: \nMean: {} Std: {} Max: {} Min: {}'.format(
        accuracies.mean(), accuracies.std(), accuracies.max(), accuracies.min()))


print('-- Random Forest Classifier --------------------')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=20)
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)
print('Accuracy score: ', accuracy_score(y_test, y_pred_rfc))
print('Confusion matrix accuracy: %f'% confusion_matrix_accuracy(y_test, y_pred_rfc))

# O cross_val_score é a melhor forma de avaliar o modelo
rfc_accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print('Cross Validation Score: \nMean: {} Std: {} Max: {} Min: {}'.format(
        rfc_accuracies.mean(), rfc_accuracies.std(), rfc_accuracies.max(), rfc_accuracies.min()))
          
# O reultado médio foi Mean: 0.2920307715669243

from sklearn.model_selection import GridSearchCV

parameters = {'max_features':('sqrt', 'log2'), 
              'n_estimators':[20, 30]}

grid_search = GridSearchCV( estimator = rfc,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 10, #cross validation
                           n_jobs = 4)

grid_search.fit(X_train, y_train)

best_accuracies = grid_search.best_score_
best_parameters = grid_search.best_params_

# O melhor foi 0.2953935541022902

print('-- Logistic Regression --------------------')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print('Accuracy score: ', accuracy_score(y_test, y_pred_lr))
print('Confusion matrix accuracy: %f'% confusion_matrix_accuracy(y_test, y_pred_lr))

print('-- K-nearest Neighbors Classifier --------------------')
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train) 
y_pred_neigh = lr.predict(X_test)

#print(classification_report(y_test, y_pred_lr))
print('Accuracy score: ', accuracy_score(y_test, y_pred_neigh))
print('Confusion matrix accuracy: %f'% confusion_matrix_accuracy(y_test, y_pred_neigh))



