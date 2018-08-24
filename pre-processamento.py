#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Recuperar os Dados

df = pd.read_csv('./data/titanic-train.csv')

df.head()
df.info()
df.describe()

df.iloc[3]
df['Ticket'].head()
df[['Embarked','Survived']].head()

# Selecionar

df[(df['Age'] > 50) | (df['Survived'] == 1)]


# Ordenação

df.sort_values('Age', ascending = True).head()

# Contagem e Agregação

df['Survived'].value_counts()

df['PassengerId'].value_counts()

df.groupby(['Pclass','Survived'])['PassengerId'].count()

# Correlações

df['IsFemale'] = df['Sex'] == 'female'

correlacao_sobrevivente = df.corr()['Survived'].sort_values()

# Removendo

df.drop(['PassengerId'],axis=1,inplace=True)

# Numpy e Visualização

dados1 = np.random.normal(0,0.1,1000)

dados2 = np.random.normal(1,0.4,1000) + np.linspace(0, 1, 1000)

dados3 = 2 + np.random.random(1000) * np.linspace(1,5,1000)

dados = np.vstack([dados1,dados2,dados3]).transpose()

df2 = pd.DataFrame(dados, columns =['Dados1','Dados2','Dados3',])

df2.describe()

# Gráfico de Linha

df2.plot(title='Grafico de Linha')

plt.plot(df2)
plt.legend(['Dados 1','Dados 2','Dados 3'])


# Gráfico de Dispersão

df2.plot(style='.')

# Histograma

df2.plot(kind='hist',bins=50,title='Histograma',alpha=0.5,normed=True,cumulative=True)

# gráfico de Caixa

df2.plot(kind='box')

# Gráfico de Pizza

pedaco = df2['Dados1'] > 0.1

quantidadefalsospositivos = pedaco.value_counts()

quantidadefalsospositivos.plot(kind='pie')

# Imagens

from PIL import Image

img = Image.open('data/dog1.jpg')

img

imgarray = np.asarray(img)

imgarray.shape

# Som

from scipy.io import wavfile

taxa, snd = wavfile.read(filename='../data/sms.wav')

from IPython.display import Audio

Audio(data=snd,rate=taxa)

plt.plot(snd)
plt.ylabel('Frequência (Hz)')
plt.xlabel('Tempo (s)')














