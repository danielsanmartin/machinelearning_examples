#!/usr/bin/env python3

# Regressão linear simples

import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('./data/50_Startups.csv')


X = dataset.iloc[:,:-1].values # todas as linhas menos a última coluna
y = dataset.iloc[:,:1].values # última coluna


# Transforma a variável categorica em numérica
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelec = LabelEncoder()
onehot = OneHotEncoder(categorical_features = [3])

X[:,3] = labelec.fit_transform(X[:,3])

X = onehot.fit_transform(X).toarray()


# Feature scaling
# a ideia é deixar todas as variáveis na mesma escala.

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.25, random_state = 0)

from sklearn.linear_model import LinearRegression

preditor = LinearRegression()
preditor.fit(X_treino, y_treino)

preditor.score(X_teste, y_teste)

for coeficiente in preditor.coef_:
    print(coeficiente)
    
print('%f'%preditor.intercept_)

y_predito = preditor.predict(X_teste)

from sklearn.metrics import r2_score

r2_score(y_teste, y_predito)

plt.scatter(X_treino, y_treino, color='red')
plt.plot(X_treino, preditor.predict(X_treino), color='blue')
plt.title('Salário x Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()