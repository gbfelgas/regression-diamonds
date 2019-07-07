#!/usr/bin/env python
# coding: utf-8

# # Trabalho 1 - Regressão Multivariável
# ## Estimativa de preços de diamantes de acordo com suas características.
# 
# UFRJ/POLI/DEL - Introdução ao Aprendizado de Máquina (EEL891) <br>
# Prof. Heraldo Almeira - Julho de 2019 <br>
# Maria Gabriella Andrade Felgas (DRE: 111471809)

# ### Importando as Bibliotecas e Ferramentas

# In[1]:


# Importando as bibliotecas e setando o ambiente de desenvolvimento

# Bibliotecas para processamento e manipulacao dos dados
import numpy as np
import pandas as pd

# Bibliotecas para visualizacao dos dados
import matplotlib.pyplot as plt
import seaborn as sns


# Bibliotecas dos modelos de treinamento
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor,                             GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Bibliotecas de ferramentas e métricas
from sklearn.preprocessing import Imputer, Normalizer, scale, MinMaxScaler, StandardScaler, Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error 


# ### Análise e Tratamento de Dados

# #### Importação de Dados

# In[2]:


# Carregando os dados de treino como dataframe
# e observando os atributos

train = pd.read_csv('data/train.csv')
train.head()


# In[3]:


train.shape


# In[4]:


train.info()


# In[5]:


# Setando o index do arquivo como arquivo do dataframe

train = train.set_index('id')
train.head()


# In[6]:


# Carregando os dados de teste como dataframe

test = pd.read_csv('data/test.csv')
test.head()


# In[7]:


test.shape


# In[8]:


test.info()


# In[9]:


# Setando o index do arquivo como arquivo do dataframe

test = test.set_index('id')
test.head()


# In[10]:


# Verificando se existem valores nulos para o conjunto de treino

train.isnull().sum()


# In[11]:


# Verificando se existem valores nulos para o conjunto de teste

test.isnull().sum()


# In[12]:


# Verificando os detalhes de cada caracteristica

train.describe()


# In[13]:


# Verificando os detalhes de cada caracteristica

test.describe()


# Como x, y e z são variáveis relacionadas às dimensões de cada diamante, não faz sentido que nenhuma delas seja igual a 0. Assim, é necessário retirar estes dados do conjunto de treino para que o modelo não seja prejudicado.

# In[14]:


# Para realizar este processamento, redefine-se o conjunto de treino como o que corresponde a condicao a seguir

train = train[(train[['x','y','z']] != 0).all(axis=1)]

# Para confirmar
train.describe()


# Agora, verifica-se a distribuição de cada um dos atributos numéricos do dataset, verificando seus padrões e outliers.

# #### Tratamento dos Dados

# ##### Preço

# ###### Observação

# In[15]:


# Analisando as caracteristicas do atributo

train['price'].describe()


# In[16]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['price'].skew()))
print('Kurtosis: {}'.format(train['price'].kurt()))

# Plotando o histograma

train['price'].hist(bins = 200)
plt.show()


# ###### Removendo os Outliers

# Como o preço é o alvo a ser considerado no treinamento do modelo, decidi manter todos os dados por enquanto.

# ##### Carat

# ###### Observação

# In[17]:


# Analisando as caracteristicas do atributo

train['carat'].describe()


# In[18]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['carat'].skew()))
print('Kurtosis: {}'.format(train['carat'].kurt()))


# Plotando o histograma

train['carat'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'carat', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com carat acima de 3.

# In[19]:


train = train[train['carat'] < 3]

# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente apos remocao

print('Skewness: {}'.format(train['carat'].skew()))
print('Kurtosis: {}'.format(train['carat'].kurt()))

# Plotando o diagrama de dispersão novamente

train.plot.scatter(x = 'carat', y = 'price')
plt.show()


# ##### x

# ###### 2.2.3.1. Observação

# In[20]:


# Analisando as caracteristicas do atributo

train['x'].describe()


# In[21]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['x'].skew()))
print('Kurtosis: {}'.format(train['x'].kurt()))

# Plotando o histograma

train['x'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'x', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, não é necessário remover outliers para este atributo.

# ##### y

# ###### Observação

# In[22]:


# Analisando as caracteristicas do atributo

train['y'].describe()


# In[23]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['y'].skew()))
print('Kurtosis: {}'.format(train['y'].kurt()))

# Plotando o histograma

train['y'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'y', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com y acima de 10.

# In[24]:


train = train[train['y'] < 10]

# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente apos remocao

print('Skewness: {}'.format(train['y'].skew()))
print('Kurtosis: {}'.format(train['y'].kurt()))

# Plotando o diagrama de dispersão novamente

train.plot.scatter(x = 'y', y = 'price')
plt.show()


# ##### z

# ###### Observação

# In[25]:


# Analisando as caracteristicas do atributo

train['z'].describe()


# In[26]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['z'].skew()))
print('Kurtosis: {}'.format(train['z'].kurt()))

# Plotando o histograma

train['z'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'z', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com z abaixo de 2 e acima de 5.5.

# In[27]:


train = train[train['z'] > 2]
train = train[train['z'] < 5.5]

# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente apos remocao

print('Skewness: {}'.format(train['z'].skew()))
print('Kurtosis: {}'.format(train['z'].kurt()))

# Plotando o diagrama de dispersão novamente

train.plot.scatter(x = 'z', y = 'price')
plt.show()


# ##### depth

# ###### Observação

# In[28]:


# Analisando as caracteristicas do atributo

train['depth'].describe()


# In[29]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['depth'].skew()))
print('Kurtosis: {}'.format(train['depth'].kurt()))

# Plotando o histograma

train['depth'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'depth', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com depth entre 56 e 67.

# In[30]:


train = train[train['depth'] > 56]
train = train[train['depth'] < 67]

# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente apos remocao

print('Skewness: {}'.format(train['depth'].skew()))
print('Kurtosis: {}'.format(train['depth'].kurt()))

# Plotando o diagrama de dispersão novamente

train.plot.scatter(x = 'depth', y = 'price')
plt.show()


# ##### table

# ###### Observação

# In[31]:


# Analisando as caracteristicas do atributo

train['table'].describe()


# In[32]:


# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente

print('Skewness: {}'.format(train['table'].skew()))
print('Kurtosis: {}'.format(train['table'].kurt()))

# Plotando o histograma

train['table'].hist(bins = 200)
plt.show() 


# Plotando o diagrama de dispersão

train.plot.scatter(x = 'table', y = 'price')
plt.show()


# ###### Removendo os Outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com table entre 51 e 67.

# In[33]:


train = train[train['table'] > 51]
train = train[train['table'] < 67]

# Definindo os valores de Skewness e Kurtosis para analisar
# a simetria e quantidade de outliers respectivamente apos remocao

print('Skewness: {}'.format(train['table'].skew()))
print('Kurtosis: {}'.format(train['table'].kurt()))

# Plotando o diagrama de dispersão novamente

train.plot.scatter(x = 'table', y = 'price')
plt.show()


# In[34]:


# Cria a matriz de correlacao entre os atributos
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# Os valores de x, y, z estão bastante correlacionados entre si e tem bastante influencia sobre o preço. Desta forma, eles foram transformados em um único atributo, volume, que é um relação entre as três variáveis.

# In[35]:


train['volume'] = train['x'] * train['y'] * train['z']

# Cria a matriz de correlacao entre os atributos, agora com a adicao do atributo volume
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# Como pode ser observado, os atributos carat e volume tem correlação de 1, o que faz sentido, considerando que carat representa o peso em quilates de cada diamante. Será que faz sentido manter ambos os atributos?
# 
# Decidiu-se testar uma nova combinação entre x e y como demonstrado a seguir.

# In[36]:


train['ratio'] = train['x'] / train['y']

# Cria a matriz de correlacao entre os atributos, agora com a adicao do atributo ratio
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# Como o novo atributo testado não parece influenciar suficientemente o preço e o atributo de volume possui correlação unitária com carat, decidi retornar o conjunto de dados ao seu estado original. 

# In[37]:


train = train.drop(['volume'], axis = 1)
train = train.drop(['ratio'], axis = 1)

train.head()


# In[38]:


# Relacionando os atributos literais ao preco, com visualizacao

# Analisando a influencia de cut
sns.barplot(x = "price", y = "cut", data = train)
plt.show()

# Analisando a influencia de color
sns.barplot(x = "price", y = "color", data = train)
plt.show()

# Analisando a influencia de clarity
sns.barplot(x = "price", y = "clarity", data = train)
plt.show()


# Como pode ser observado nos gráficos acima, as opções de cada um dos atributos influenciam o preço de maneiras diferentes. As barras coloridas significam o valor estimado para cada opção e a linha ao final de cada barra informa a incerteza destas estimativas.

# Por conta deste fenomeno, decidiu-se lidar com estes atributos de duas maneiras diferentes:
# - Transformando cada uma destas opções em novos atributos;
# - Atribuindo valores numéricos de acordo com a influência sobre o preço, valores maiores para aqueles que tem valores estimados de preço maiores.
#     
# A primeira opção é realizada desta forma pois os valores numéricos influenciam diretamente os modelos de regressão que utilizam os pesos relacionados a cada atributo, como a regressão linear.
# A segunda opção transforma cada sub-característica em um atributo booleano e pode influenciar positivamente na maneira como os modelos funcionam.

# In[39]:


# Criando novos atributos a partir das classes dos atributos literais para o conjunto de treino

# train_1 eh referente a primeira alternativa de tratamento dos atributos literais
train_1 = pd.get_dummies(train)
train_1.head()


# In[40]:


# Criando novos atributos a partir das classes dos atributos literais para o conjunto de teste

# test_1 eh referente a primeira alternativa de tratamento dos atributos literais
test_1 = pd.get_dummies(test)
test_1.head()


# In[41]:


# Criando a matriz de correlacao para o primeiro tratamento
corr_matrix_1 = train_1.corr()

plt.subplots(figsize = (25, 25))
sns.heatmap(corr_matrix_1, square=True, cbar=True, annot = True, cmap='GnBu')
plt.show()


# In[42]:


# Substituindo os valores dos atributos de acordo com a observacao acima para o conjunto de treino

# train_2 eh referente a segunda alternativa de tratamento dos atributos literais
train_2 = train.copy()

# cut
train_2['cut'] = train_2['cut'].replace({'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4, 'Premium': 5}, inplace = False)

# color
train_2['color'] = train_2['color'].replace({'E': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}, inplace = False)

# clarity
train_2['clarity'] = train_2['clarity'].replace({'VVS1': 1, 'IF': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5,                                                  'SI1': 6, 'I1': 7, 'SI2': 8}, inplace = False)

train_2.head()


# In[43]:


# Substituindo os valores dos atributos de acordo com a observacao acima para o conjunto de teste

# test_2 eh referente a segunda alternativa de tratamento dos atributos literais
test_2 = test.copy()

# cut
test_2['cut'] = test_2['cut'].replace({'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4, 'Premium': 5}, inplace = False)

# color
test_2['color'] = test_2['color'].replace({'E': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}, inplace = False)

# clarity
test_2['clarity'] = test_2['clarity'].replace({'VVS1': 1, 'IF': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5,                                                  'SI1': 6, 'I1': 7, 'SI2': 8}, inplace = False)

test_2.head()


# In[44]:


# Criando a matriz de correlacao para o segundo tratamento
corr_matrix_2 = train_2.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix_2, square=True, cbar=True, annot = True, cmap='GnBu')
plt.show()


# In[45]:


# Criando o conjunto de treino e de teste a partir do conjunto da primeira alternativa

x_1 = train_1.drop(['price'], axis = 1)
y_1 = train_1['price']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, random_state = 2, test_size=0.3)


# In[46]:


# Verificando o conjunto de treino

X_train_1.head()


# In[47]:


# Verificando o conjunto de teste

X_test_1.head()


# In[48]:


# Criando o conjunto de treino e de teste a partir da segunda alternativa

x_2 = train_2.drop(['price'], axis = 1)
y_2 = train_2['price']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, random_state = 2, test_size=0.3)


# In[49]:


# Verificando o conjunto de treino

X_train_2.head()


# In[50]:


# Verificando o conjunto de teste

X_test_2.head()


# Para definir quais são os atributos mais importantes, aplica-se um algoritmo de Random Forest que, através do objeto SelectFromModel, seleciona aqueles que possuem maior peso sobre o preço.

# In[51]:


# Definindo a funcao que calcula o RMSPE para comparar os erros

def rmspe_score(y_test, y_pred):

    return np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis = 0))

# Inicializando listas para guardar os scores
R2_Scores_1 = []
R2_Scores_2 = []
models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 'Ridge Regression',          'GradientBoosting Regression', 'RandomForest Regression', 'KNeighbours Regression']


# #### Linear

# In[52]:


# Definindo a funcao de treinamento do modelo de regressão linear
def train_lr(X_train, y_train, X_test, y_test, data):
    lr = LinearRegression()
    lr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 5,verbose = 1)
    y_pred = lr.predict(X_test)
    print('')
    print('####### Linear Regression #######')
    print('Score : %.4f' % lr.score(X_test, y_test))
    print(accuracies)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    rmspe = rmspe_score(y_test, y_pred)

    print('')
    print('MSE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mse)
    print('MAE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mae)
    print('RMSE  - Conjunto de dados {}'.format(data),' : %0.2f ' % rmse)
    print('R2    - Conjunto de dados {}'.format(data),' : %0.2f ' % r2)
    print('RMSPE - Conjunto de dados {}'.format(data),' : %0.2f ' % rmspe)
    
    if data == 1:
        R2_Scores_1.append(r2)
    else:        
        R2_Scores_2.append(r2)


# In[53]:


# Treinando linear para o primeiro conjunto de dados

train_lr(X_train_1, y_train_1, X_test_1, y_test_1, data = 1)


# In[54]:


# Treinando linear para o segundo conjunto de dados

train_lr(X_train_2, y_train_2, X_test_2, y_test_2, data = 2)


# #### Lasso

# In[55]:


def train_la(X_train, y_train, X_test, y_test, data):
    la = Lasso(normalize=True)
    la.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = la, X = X_train, y = y_train, cv = 5,verbose = 1)
    y_pred = la.predict(X_test)
    print('')
    print('###### Lasso Regression ######')
    print('Score : %.4f' % la.score(X_test, y_test))
    print(accuracies)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    rmspe = rmspe_score(y_test, y_pred)

    print('')
    print('MSE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mse)
    print('MAE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mae)
    print('RMSE  - Conjunto de dados {}'.format(data),' : %0.2f ' % rmse)
    print('R2    - Conjunto de dados {}'.format(data),' : %0.2f ' % r2)
    print('RMSPE - Conjunto de dados {}'.format(data),' : %0.2f ' % rmspe)
    
    if data == 1:
        R2_Scores_1.append(r2)
    else:        
        R2_Scores_2.append(r2)


# In[56]:


# Treinando lasso para o primeiro conjunto de dados

train_la(X_train_1, y_train_1, X_test_1, y_test_1, data = 1)


# In[57]:


# Treinando lasso para o segundo conjunto de dados

train_la(X_train_2, y_train_2, X_test_2, y_test_2, data = 2)


# #### AdaBoost

# In[58]:


def train_ar(X_train, y_train, X_test, y_test, data):
    ar = AdaBoostRegressor(n_estimators=1000)
    ar.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = ar, X = X_train, y = y_train, cv = 5,verbose = 1)
    y_pred = ar.predict(X_test)
    print('')
    print('###### AdaBoost Regression ######')
    print('Score : %.4f' % ar.score(X_test, y_test))
    print(accuracies)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    rmspe = rmspe_score(y_test, y_pred)

    print('')
    print('MSE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mse)
    print('MAE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mae)
    print('RMSE  - Conjunto de dados {}'.format(data),' : %0.2f ' % rmse)
    print('R2    - Conjunto de dados {}'.format(data),' : %0.2f ' % r2)
    print('RMSPE - Conjunto de dados {}'.format(data),' : %0.2f ' % rmspe)


# In[59]:


# Treinando adaboost para o primeiro conjunto de dados

train_ar(X_train_1, y_train_1, X_test_1, y_test_1, data = 1)


# In[60]:


# Treinando adaboost para o segundo conjunto de dados

train_ar(X_train_2, y_train_2, X_test_2, y_test_2, data = 2)


# #### Random Forest

# In[61]:


def train_rf(X_train, y_train, X_test, y_test, data):
    rf = RandomForestRegressor(random_state = 2)
    rf.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 5,verbose = 1)
    y_pred = rf.predict(X_test)
    print('')
    print('###### Random Forest ######')
    print('Score : %.4f' % rf.score(X_test, y_test))
    print(accuracies)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    rmspe = rmspe_score(y_test, y_pred)

    print('')
    print('MSE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mse)
    print('MAE   - Conjunto de dados {}'.format(data),' : %0.2f ' % mae)
    print('RMSE  - Conjunto de dados {}'.format(data),' : %0.2f ' % rmse)
    print('R2    - Conjunto de dados {}'.format(data),' : %0.4f ' % r2)
    print('RMSPE - Conjunto de dados {}'.format(data),' : %0.4f ' % rmspe)


# In[62]:


# Treinando random forest para o primeiro conjunto de dados

train_rf(X_train_1, y_train_1, X_test_1, y_test_1, data = 1)


# In[63]:


# Treinando random forest para o segundo conjunto de dados

train_rf(X_train_2, y_train_2, X_test_2, y_test_2, data = 2)


# In[64]:


x = train_2.drop(['price'], axis = 1)
y = train_2['price']

rf = RandomForestRegressor(random_state = 2)
rf.fit(x, y)
y_pred = rf.predict(test_2)

submission = pd.DataFrame({'id':test_2.index, 'price':y_pred})
submission.head()

submission.to_csv('data/submission.csv', index = False)


# In[ ]:




