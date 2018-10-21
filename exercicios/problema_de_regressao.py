#!/usr/bin/env python3
# Autor: Daniel San Martin
# Data: 16/09/2018
# Previsao dos precos dos diamantes
"""
1) Problema de Regressão


Descobrir o melhor algoritmo para prever o preço de dimantes, dado algumas características. Seguem as descrições das características do dataset "diamonds.zip":

price: de preço em dólares americanos (\ $ 326 - \ $ 18,823)

carat: peso em quilate do diamante (0,2--5,01)

cut: qualidade de corte do corte (Justo, Bom, Muito Bom, Premium, Ideal)

color: cor de diamante, de J (pior) a D (melhor)

clarity: clareza a medição da clareza do diamante (I1 (pior), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (melhor))

x: comprimento em mm (0-10.74)

y: largura em mm (0--58,9)

z: profundidade de z em mm (0--31.8)

depth: profundidade total percentual = z / média (x, y) = 2 * z / (x + y) (43--79)

table: largura da mesa do topo do diamante em relação ao ponto mais largo (43--95)


Além dos algoritmos de regressão vistos, utilizam a regularização para regressão linear com as APIs Ridge e Lasso  da biblioteca sklearn.

1) http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

2) http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

Procurem os melhores parâmetros para cada algoritmo.


"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('./diamonds.csv')

print(dataset.head())
# verifico se o dataset tem valores nulos
print(dataset.info())
#print(dataset.describe())

del dataset['Unnamed: 0'] #removo o que parece ser uma coluna de Ids
print(dataset.head())

# Verificando a correlacao par a par dos valores
print(dataset.corr())

# a variavel depth possui baixa correlacao com o conjunto, entao a eliminamos.
del dataset['depth']

# separando os dados
X = dataset.drop(['price'],axis=1) # variaveis independentes

y = dataset['price'] # variaves dependente

# verifico os valores unicos das colunas categoricas
#print(dataset.groupby('cut')['price'].mean().sort_values())
#print(dataset.groupby('color')['price'].mean().sort_values())
#print(dataset.groupby('clarity')['price'].mean().sort_values())

"""
Segundo as informacoes sobre os dados:
    
cut (qualidade de corte do corte) em ordem (Fair, Good, Very Good, Premium, Ideal)
color (cor do diamente) vai do J (pior) a D (melhor)
clareza a medição da clareza do diamante (I1 (pior), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (melhor))

Mas olhando os prints das medias dos dados, percebe-se que existe algum problema
ja que o preco medio dos diamentes piores (Fair) esta mais alto do que os muito
bons (Very Good). O peso do diamente é uma das variáveis que notadamente mais 
influencia no preço, logo poderia ser usada para amenizar esse problema.


cut
Ideal        3457.541970
Good         3928.864452
Very Good    3981.759891
Fair         4358.757764
Premium      4584.257704
Name: price, dtype: float64
color
E    3076.752475
D    3169.954096
F    3724.886397
G    3999.135671
H    4486.669196
I    5091.874954
J    5323.818020
Name: price, dtype: float64
clarity
VVS1    2523.114637
IF      2864.839106
VVS2    3283.737071
VS1     3839.455391
I1      3924.168691
VS2     3924.989395
SI1     3996.001148
SI2     5063.028606
Name: price, dtype: float64

"""



# Transforma a variável categorica em numérica
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelec = LabelEncoder()

X['cut'] = labelec.fit_transform(X['cut'])
X['color'] = labelec.fit_transform(X['color'])
X['clarity'] = labelec.fit_transform(X['clarity'])

onehot = OneHotEncoder(categorical_features = [1,2,3])
X = onehot.fit_transform(X).toarray()


# Feature scaling
# a ideia é deixar todas as variáveis na mesma escala.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.25, random_state = 0)

################################################
# a seguir são avaliados os três modelos
################################################


################################################
# Regressao Linear

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_treino, y_treino)

print('Avaliando o LinearRegression...')
score = lr.score(X_teste, y_teste)
print('Score: ', score)

#for coeficiente in preditor.coef_:
#    print(coeficiente)

print("Intercepta: %f" %(lr.intercept_))
y_predito = lr.predict(X_teste)
rmse=np.sqrt(mean_squared_error(y_teste,y_predito))
print("Erro médio quadrático: %f" %(rmse))


################################################
# Ridge

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, random_state=None, solver='auto', tol=0.001)
clf.fit(X_treino, y_treino)

print('Avaliando o Ridge...')
score = clf.score(X_teste, y_teste)
print('Score: ', score)

print("Intercepta: %f" %(clf.intercept_))
y_predito = clf.predict(X_teste)
rmse=np.sqrt(mean_squared_error(y_teste,y_predito))
print("Erro médio quadrático: %f" %(rmse))

################################################
# Lasso

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.001, warm_start=False)
lasso.fit(X_treino, y_treino)

print('Avaliando o Lasso...')
score = lasso.score(X_teste, y_teste)
print('Score: ', score)

print("Intercepta: %f" %(lasso.intercept_))
y_predito = lasso.predict(X_teste)
rmse=np.sqrt(mean_squared_error(y_teste,y_predito))
print("Erro médio quadrático: %f" %(rmse))


"""
Escolha do melhor algoritmo

O algoritmo Ridge apresentou menor erro médio quadrático e maior score.

------------------------------------------------------
Algoritmo        | Score                 | EMQ
------------------------------------------------------
Ridge	         | ﻿0.9203859794981648	 | ﻿1127.631508
﻿Lasso	         ﻿| 0.9203832543294973	﻿ | 1127.650808
﻿LinearRegression | ﻿0.9203806692136428	﻿ | 1127.669115
------------------------------------------------------

"""

