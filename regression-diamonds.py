#!/usr/bin/env python
# coding: utf-8

# # Trabalho 1 - Regressão Multivariável
# # Estimativa de preços de diamantes de acordo com suas características.
# 
# UFRJ/POLI/DEL - Introdução ao Aprendizado de Máquina (EEL891) <br>
# Prof. Heraldo Almeira - Julho de 2019 <br>
# Maria Gabriella Andrade Felgas (DRE: 111471809)

# ## Mineração e Análise de Dados

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor,                             GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Bibliotecas de ferramentas e métricas
from sklearn.preprocessing import Imputer, Normalizer, scale, MinMaxScaler, StandardScaler, Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error 


# ### Análise e Tratamento de Dados

# ### Carregando o conjunto de treino

# In[2]:


# Carregando os dados de treino como dataframe
# e observando os atributos
train = pd.read_csv('data/train.csv')
train.head()


# In[3]:


# Verificando tamanho do dataframe
train.shape


# In[4]:


# Verificando informacoes especificas
train.info()


# In[5]:


# Setando o index do arquivo como arquivo do dataframe
train = train.set_index('id')
train.head()


# In[6]:


# Verificando se existem valores nulos para o conjunto de treino
train.isnull().sum()


# In[7]:


# Verificando os detalhes de cada caracteristica
train.describe()


# Como x, y e z são variáveis relacionadas às dimensões de cada diamante, não faz sentido que nenhuma delas seja igual a 0. Assim, é necessário retirar estes dados do conjunto de treino para que o modelo não seja prejudicado.

# In[8]:


# Para realizar este processamento, redefine-se o conjunto de treino como o que corresponde a condicao a seguir
train = train[(train[['x','y','z']] != 0).all(axis=1)]

# Para confirmar
train.describe()


# ### Carregando o conjunto de teste

# In[9]:


# Carregando os dados de teste como dataframe
test = pd.read_csv('data/test.csv')
test.head()


# In[10]:


test.shape


# In[11]:


test.info()


# In[12]:


# Setando o index do arquivo como arquivo do dataframe
test = test.set_index('id')
test.head()


# In[13]:


# Verificando se existem valores nulos para o conjunto de teste
test.isnull().sum()


# In[14]:


# Verificando os detalhes de cada caracteristica
test.describe()


# Agora, verifica-se a distribuição de cada um dos atributos numéricos do dataset, verificando seus padrões e outliers.

# ### Tratamento dos dados

# In[15]:


# Cria a matriz de correlacao entre os atributos numericos para visualizacao inicial
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# In[16]:


# Funcao de analise de atributo
def analysis(feature, hist=True):
    
    # Definindo os valores de Skewness e Kurtosis para analisar
    # a simetria e quantidade de outliers respectivamente
    print('Skewness: {}'.format(train[feature].skew()))
    print('Kurtosis: {}'.format(train[feature].kurt()))

    if hist:
        # Plotando o histograma
        plt.figure(figsize=(20,10))
        train[feature].hist(bins = 500)
        plt.show()
    
    if feature != 'price':
        # Plotando o diagrama de dispersão
        plt.figure(figsize=(20,10))
        train.plot.scatter(x = feature, y = 'price')
        plt.show()
        
        
# Funcao que checa a contagem para cada limiar    
def count_limit(feature, inf_limit, sup_limit, hop):
    
    n = int((sup_limit - inf_limit) / hop)
    p = np.zeros(((n + 1), 2))
    
    for i in range(n + 1):
        p[i][0] = train[train[feature] < (inf_limit + (hop * i))][feature].count()
        p[i][1] = np.round((p[i][0] / train[train[feature] < sup_limit][feature].count()) * 100, 2)
        print('Quantidade de pontos abaixo de {} :'.format(inf_limit + (hop * i)), p[i][0],               'Porcentagem: {} %'.format(p[i][1]))


# ### Preço

# In[17]:


train['price'].describe()


# In[18]:


analysis('price')


# In[19]:


count_limit('price', 2500, 20000, 2500)


# #### Removendo os outliers

# De acordo com os resultados acima, decidi remover os dados com preço acima de 10000.

# In[20]:


# train = train[train['price'] < 10000]

# analysis('price')


# ### Carat

# In[21]:


train['carat'].describe()


# In[22]:


analysis('carat')


# In[23]:


count_limit('carat', 0.5, 3, 0.5)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com *carat* acima de 2. Além disso, é importante observar o comportamento espaçado do atributo, assumindo conjuntos de valores a partir de determinadas "linhas bem definidas". Por conta desta característica, divido o atributo em 4 novos atributos diferentes.

# In[24]:


train = train[train['carat'] < 2.5]

analysis('carat', hist=False)


# ### x

# In[25]:


train['x'].describe()


# In[26]:


analysis('x')


# In[27]:


count_limit('x', 4, 9, 1)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima e da distribuição de *x*, não é necessário remover outliers para este atributo.

# ### y

# In[28]:


train['y'].describe()


# In[29]:


analysis('y')


# In[30]:


count_limit('y', 5, 35, 5)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima e da distribuição de *y*, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com y acima de 10.

# In[31]:


train = train[train['y'] < 10]

analysis('y')


# ### z

# In[32]:


train['z'].describe()


# In[33]:


analysis('z')


# In[34]:


count_limit('z', 1, 6, 0.5)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com z abaixo de 2.3 e acima de 5.

# In[35]:


train = train[train['z'] > 2.2]
train = train[train['z'] < 5.3]

analysis('z')


# ### depth

# In[36]:


train['depth'].describe()


# In[37]:


analysis('depth')


# In[38]:


count_limit('depth', 45, 80, 5)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com depth entre 56 e 67.

# In[39]:


train = train[train['depth'] > 56]
train = train[train['depth'] < 67]

analysis('depth')


# ### table

# In[40]:


train['table'].describe()


# In[41]:


analysis('table')


# In[42]:


count_limit('table', 45, 75, 5)


# #### Removendo os outliers

# De acordo com a observação do gráfico acima, defino os outliers como sendo os pontos fora da distribuição padrão, ou seja, com table entre 54 e 67.

# In[43]:


train = train[train['table'] > 54]
train = train[train['table'] < 67]

analysis('table')


# ### Outras observações

# In[44]:


# Criando os graficos de dispersao para visualizacao geral

sns.pairplot(train)


# In[45]:


# Cria a matriz de correlacao entre os atributos numericos
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# In[46]:


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

# Por conta deste fenômeno, decidiu-se lidar com estes atributos da seguinte maneira:
# - Atribuindo valores numéricos de acordo com a influência sobre o preço, ou seja, de acordo com o valor esperado de preço para cada categoria, através da média.
# 
# Esta opção foi escolhida pois apresentou melhores resultados em comparação com a utilização de *Hot Encoding* e está descrita pela função abaixo:

# In[47]:


# Funcao de transformacao dos atributos categoricos
def categ_feature(feature, data):
    mean = train.groupby(feature)['price'].mean()
    mean_sort = mean.reset_index().sort_values(['price']).set_index([feature]).astype(int)
    
    mean_sort.to_dict()
    mean_sort = mean_sort['price']
    
    data[feature] = data[feature].replace(mean_sort, inplace = False)
    
    return mean_sort, data


# In[48]:


# Aplicando a funcao para os dados de treino e teste
mean_sort_cut, train = categ_feature('cut', train)
test['cut'] = test['cut'].replace(mean_sort_cut, inplace = False)

mean_sort_color, train = categ_feature('color', train)
test['color'] = test['color'].replace(mean_sort_color, inplace = False)

mean_sort_clarity, train = categ_feature('clarity', train)
test['clarity'] = test['clarity'].replace(mean_sort_clarity, inplace = False)

test.head()


# In[49]:


# Criando a matriz de correlacao para o segundo tratamento
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# In[50]:


# Criando o conjunto de treino e de teste para treinar o modelo a partir de train modificado
x = train.drop(['price'], axis = 1)
y = train['price']


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 2, test_size=0.3)


# In[51]:


# Arrays de referência para comparação entre modelos
model_dict = {'Linear Regressor': 1, 'Lasso Regression': 1, 'Ridge Regression': 1, 'AdaBoost Regression': 1,             'Gradient Boosting Regression': 1, 'Random Forest Regression': 1, 'Extra Trees Regression': 1}


# In[52]:


# Função que calcula o RMSPE para validacao dos modelos
def rmspe_score(y_test, y_pred):
    
    rmspe = np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis = 0))

    return rmspe


# In[53]:


# Funcao de regressao generica, para varios modelos diferentes
def model_analysis(X_train, X_test, y_train, y_test, regressor, name):
    regressor.fit(X_train, y_train)
    accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5,verbose = 1)
    y_pred = regressor.predict(X_test)
    print('')
    print('###### {} ######'. format(name))
    print('Score : %.6f' % regressor.score(X_test, y_test))
    print(accuracies)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    rmspe = rmspe_score(y_test, y_pred)

    print('')
    print('MSE   : %0.6f ' % mse)
    print('MAE   : %0.6f ' % mae)
    print('RMSE  : %0.6f ' % rmse)
    print('R2    : %0.6f ' % r2)
    print('RMSPE : %0.6f ' % rmspe)
    
    model_dict[name] = round(rmspe, 6)


# In[54]:


get_ipython().run_cell_magic('time', '', "\nlr = LinearRegression()\nmodel_analysis(X_train, X_test, y_train, y_test, lr, 'Linear Regressor')")


# In[55]:


get_ipython().run_cell_magic('time', '', "\nlar = Lasso(normalize = True)\nmodel_analysis(X_train, X_test, y_train, y_test, lar, 'Lasso Regression')")


# In[56]:


get_ipython().run_cell_magic('time', '', "\nrr = Ridge(normalize = True)\nmodel_analysis(X_train, X_test, y_train, y_test, rr, 'Ridge Regression')")


# In[57]:


get_ipython().run_cell_magic('time', '', "\nabr = AdaBoostRegressor(random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, abr, 'AdaBoost Regression')")


# In[63]:


get_ipython().run_cell_magic('time', '', "\ngbr = GradientBoostingRegressor(n_estimators = 200, min_samples_leaf = 2, min_samples_split = 5, \\\n                                max_depth = 10, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, gbr, 'Gradient Boosting Regression')")


# In[64]:


get_ipython().run_cell_magic('time', '', "\nrfr = RandomForestRegressor(n_estimators = 250, n_jobs = 2, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, rfr, 'Random Forest Regression')")


# In[65]:


get_ipython().run_cell_magic('time', '', "\netr = ExtraTreesRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, etr, 'Extra Trees Regression')")


# In[66]:


compare = pd.DataFrame()
compare['Model'] = model_dict.keys()
compare['RMSPE'] = model_dict.values()

compare = compare.set_index('Model').sort_values(['RMSPE'])
compare


# In[ ]:


# Modelo com menor RMSPE eh escolhido
x = train.drop(['price'], axis = 1)
y = train['price']

gbr = GradientBoostingRegressor(n_estimators = 250, min_samples_leaf = 2, min_samples_split = 5,                                 max_depth = 10, random_state = 152)
gbr.fit(x, y)
y_pred = gbr.predict(test)

submission = pd.DataFrame({'id':test.index, 'price':y_pred})
submission.head()

# submission.to_csv('data/submission.csv', index = False)


# In[ ]:




