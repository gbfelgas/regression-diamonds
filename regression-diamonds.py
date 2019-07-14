#!/usr/bin/env python
# coding: utf-8

# # Trabalho 1 - Regressão Multivariável
# # Estimativa de preços de diamantes de acordo com suas características.
# 
# UFRJ/POLI/DEL - Introdução ao Aprendizado de Máquina (EEL891)  
# Prof. Heraldo Almeira - Julho de 2019 
# Maria Gabriella Andrade Felgas (DRE: 111471809)

# # Introdução

# Este trabalho tem como objetivo desenvolver um modelo de regressão para estimar os preços de diamantes a partir de características específicas dadas. Para realizá-lo, foram disponibilizados um conjunto de dados de treino, com alvo, e um conjunto de dados de teste, cujo alvo deve ser estimado pela aluna, além de um modelo do arquivo a ser submetido à competição.
# 
# Cada atributo do conjunto de dados está descrito abaixo:
# 
# - **id**: Identificação única do diamante;
# - **carat**: Peso em quilates (1 quilate = 0,2 g);
# - **cut**: Qualidade da lapidação, em uma escala categórica ordinal com os seguinte valores: 
# >- **"Fair"** = Aceitável (classificação de menor valor); 
# >- **"Good"** = Boa;
# >- **"Very Good"** = Muito boa;
# >- **"Premium"** = Excelente;
# >- **"Ideal"** = Perfeita (classificação de maior valor).
# - **color**: Cor, em uma escala categórica ordinal com os seguintes valores: 
# >- **"D"** = Excepcionalmente incolor extra (classificação de maior valor); 
# >- **"E"** = Excepcionalmente incolor;
# >- **"F"** = Perfeitamente incolor;
# >- **"G"** = Nitidamente incolor; 
# >- **"H"** = Incolor;
# >- **"I"** = Cor levemente perceptível;
# >- **"J"** = Cor perceptível (classificação de menor valor).
# - **clarity**: Pureza, em uma escala categórica ordinal com os seguintes valores: 
# >- **"I1"** = Inclusões evidentes com lupa de 10x (classificação de menor valor);
# >- **"SI2"** e **"SI1"** = Inclusões pequenas, mas fáceis de serem visualizadas com lupa de 10x;
# >- **"VS2"** e **"VS1"** = Inclusões muito pequenas e difíceis de serem visualizadas com lupa de 10x; 
# >- **"VVS2"** e **"VVS1"** = Inclusões extremamente pequenas e muito difíceis de serem visualizadas com lupa de 10x;
# >- **"IF"** = Livre de inclusões (classificação de maior valor).
# - **x**: Comprimento em milímetros;
# - **y**: Largura em milímetros;
# - **z**: Profundidade em milímetros;
# - **depth**: Profundidade relativa = 100 * z / mean(x,y) = 200 * z / ( x + y );
# - **table**: Razão percentual entre entre a largura no topo e a largura no ponto mais largo;
# - **price**: Preço do diamante, em dólares americanos.  
# ------------------------------------------------------------------------------------------------------------------------
# **OBS:** Este documento apresenta partes com código comentado devido aos testes realizados durante o desenvolvimento do modelo. Foram mantidos para melhor compreensão da lógica utilizada e para reprodução, a quem interessar.
# 

# # Importando as Bibliotecas e Ferramentas

# Para realizar este trabalho, foi necessário utilizar diversas bibliotecas disponíveis em Python:
# 
# - **Processamento e manipulação de dados:** Numpy e Pandas;
# - **Visualização de dados:** Matplotlib e Seaborn;
# - **Modelos de treinamento, ferramentas e métricas:** Scikit-learn.

# In[188]:


# Importando as bibliotecas e setando o ambiente de desenvolvimento

# Bibliotecas para processamento e manipulacao dos dados
import numpy as np
import pandas as pd

# Bibliotecas para visualizacao dos dados
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas dos modelos de treinamento
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

# Bibliotecas de ferramentas e métricas
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer


# # Mineração e Análise de Dados

# A seguir, segue o passo a passo para analisar e tratar o conjunto de dados de acordo com as observações.

# ### Carregando Conjunto de Treino

# In[189]:


# Carregando os dados de treino como dataframe
# e observando os atributos
train = pd.read_csv('data/train.csv')
train.head()


# In[190]:


# Verificando tamanho do dataframe
train.shape


# In[191]:


# Verificando informacoes especificas
train.info()


# In[192]:


# Setando o index do arquivo como index do dataframe
train = train.set_index('id')
train.head()


# In[193]:


# Verificando se existem valores nulos para o conjunto de treino
train.isnull().sum()


# In[194]:


# Verificando os detalhes de cada caracteristica
train.describe()


# Como x, y e z são variáveis relacionadas às dimensões de cada diamante, não faz sentido que nenhuma delas seja igual a 0. Assim, é necessário retirar estes dados do conjunto de treino para que o modelo não seja prejudicado.

# In[195]:


# Para realizar este processamento, redefine-se o conjunto de treino
# como todos os dados em que x, y e z sao diferentes de 0
train = train[(train[['x','y','z']] != 0).all(axis=1)]

# Para confirmar
train.describe()


# Como pode ser observado, agora, o conjunto de treino não possui mais x, y e z zerados, o que se confirma pelos valores encontrados na linha de mínimos de cada um desses atributos acima.

# ### Carregando Conjunto de Teste

# In[196]:


# Carregando os dados de teste como dataframe
test = pd.read_csv('data/test.csv')
test.head()


# In[197]:


test.shape


# In[198]:


test.info()


# In[199]:


# Setando o index do arquivo como index do dataframe
test = test.set_index('id')
test.head()


# In[200]:


# Verificando se existem valores nulos para o conjunto de teste
test.isnull().sum()


# In[201]:


# Verificando os detalhes de cada caracteristica
test.describe()


# Agora, verifica-se a distribuição de cada um dos atributos numéricos do dataset, verificando seus padrões e outliers.

# ### Tratamento de Dados

# In[202]:


# Cria-se a matriz de correlacao entre os atributos numericos para visualizacao inicial
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# A matriz de correlação acima possui apenas os atributos numéricos do conjunto de dados, excluindo as características categóricas descritas anteriormente.  
# 
# A partir deste gráfico, é possível observar que **carat, x, y e z** são os atributos de maior correlação com o preço e, fatalmente, maior correlação entre si também, já que descrevem características extremamente dependentes umas das outras, como peso, comprimento, largura e profundidade.

# Define-se, então, uma função de análise, para verificar a relação de assimetria e curtose na distribuição de cada atributo numérico e para observar seu histograma e diagrama de dispersão em relação ao preço, e uma função de contagem de dados até determinados limiares, configurados manualmente de acordo com o histograma.

# In[203]:


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


# #### Preço

# In[204]:


train['price'].describe()


# In[205]:


analysis('price')


# In[206]:


count_limit('price', 2500, 20000, 2500)


# ##### Removendo os Outliers

# Como o preço é o alvo a ser estimado pelo modelo, não faz sentido remover nenhum valor específico.

# #### Carat

# In[207]:


train['carat'].describe()


# In[208]:


analysis('carat')


# In[209]:


count_limit('carat', 0.5, 3, 0.5)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **carat** abaixo de 2.4, por exemplo. Além disso, é importante observar o comportamento espaçado do atributo, assumindo conjuntos de valores a partir de determinadas "linhas bem definidas".  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[210]:


# train = train[train['carat'] < 2.4]

# analysis('carat', hist=False)


# #### x

# In[211]:


train['x'].describe()


# In[212]:


analysis('x')


# In[213]:


count_limit('x', 4, 9, 1)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **x** abaixo de 9, por exemplo.  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[214]:


# train = train[train['x'] < 9]

# analysis('x', hist=False)


# #### y

# In[215]:


train['y'].describe()


# In[216]:


analysis('y')


# In[217]:


count_limit('y', 5, 35, 5)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **y** abaixo de 10, por exemplo.  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[218]:


# train = train[train['y'] < 10]

# analysis('y')


# #### z

# In[219]:


train['z'].describe()


# In[220]:


analysis('z')


# In[221]:


count_limit('z', 1, 6, 0.5)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **z** acima de 2.2 e abaixo de 5.4, por exemplo.  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[222]:


# train = train[train['z'] > 2.2]
# train = train[train['z'] < 5.4]

# analysis('z')


# #### depth

# In[223]:


train['depth'].describe()


# In[224]:


analysis('depth')


# In[225]:


count_limit('depth', 45, 80, 5)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **depth** acima de 56 e abaixo de 67, por exemplo.  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[226]:


# train = train[train['depth'] > 56]
# train = train[train['depth'] < 67]

# analysis('depth')


# #### table

# In[227]:


train['table'].describe()


# In[228]:


analysis('table')


# In[229]:


count_limit('table', 45, 75, 5)


# ##### Removendo os Outliers

# De acordo com a observação do gráfico acima e da quantidade de dados acumulados, os outliers poderiam ser considerados com **table** acima de 56 e abaixo de 67, por exemplo.  
# O modelo foi testado considerando a remoção deste dados, porém, apresentou melhores resultados considerando todos os casos, por isso, o código abaixo se encontra comentado.

# In[230]:


# train = train[train['table'] > 56]
# train = train[train['table'] < 67]

# analysis('table')


# ### Outras Observações

# Para visualizar todos os atributos numéricos e as relações entre eles, plotam-se vários gráficos de dispersão para evidenciar as dependências possíveis e, novamente, a matriz de correlações, para analisar a remoção de outliers caso ela seja aplicada.

# In[231]:


# Criando os graficos de dispersao para visualizacao geral

sns.pairplot(train)


# In[232]:


# Cria a matriz de correlacao entre os atributos numericos
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# ### Atributos Categóricos

# Para tratar os atributos categóricos, observou-se, inicialmente, a relação de cada uma das categorias com o preço e o tamanho da variância desta estimativa média.

# In[233]:


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


# Como pode ser observado nos gráficos acima, as categorias de cada um dos atributos influenciam o preço de maneiras diferentes. As barras coloridas significam o valor estimado para cada opção e a linha ao final de cada barra informa a incerteza destas estimativas.

# Por conta deste fenômeno, decidiu-se lidar com estes atributos da seguinte maneira:
# 1. Utilizando *Hot Enconding*, ou seja, dividindo cada um deles em novos atributos booleanos referentes às suas categorias através da função *get_dummies()*. Esta alternativa foi descartada e não está presente neste documento, pois todos os seus resultados apresentaram comportamento piorado em relação aos demais.
# 2. Substituindo cada uma das categorias por um valor inteiro que representasse a contribuição em relação ao preço. Ou seja, categorias com maior valor estimado de preço deveriam possuir maior valor inteiro de substituição e categorias com menor valor estimado, menor. Este procedimento foi realizado de duas maneiras diferentes:
# > 1. Considerando valores de 1 até o número de categorias de cada atributo categórico;
# > 2. Considerando as médias de preço por categoria.
# 
# As duas últimas opções foram escolhidas pois apresentaram melhores resultados e podem ser observadas abaixo:

# In[234]:


# Numeros inteiros de 1 a numero de categorias do atributo
# Valores escolhidos de acordo com observacao dos graficos

# cut
train['cut'] = train['cut'].replace({'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4, 'Premium': 5})
test['cut']  = test['cut'].replace({'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4, 'Premium': 5})

# color
train['color'] = train['color'].replace({'E': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7})
test['color']  = test['color'].replace({'E': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7})

# clarity
train['clarity'] = train['clarity'].replace({'VVS1': 1, 'IF': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5,                                              'SI1': 6, 'I1': 7, 'SI2': 8})
test['clarity']  = test['clarity'].replace({'VVS1': 1, 'IF': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5,                                             'SI1': 6, 'I1': 7, 'SI2': 8})

# Visualizando teste para checar funcionamento
test.head()


# In[235]:


# # Funcao de transformacao dos atributos categoricos
# # nas medias de preco por categoria

# def categ_feature(feature, data):
#     mean = train.groupby(feature)['price'].mean()
#     mean_sort = mean.reset_index().sort_values(['price']).set_index([feature]).astype(int)
    
#     mean_sort.to_dict()
#     mean_sort = mean_sort['price']
    
#     data[feature] = data[feature].replace(mean_sort, inplace = False)
    
#     return mean_sort, data


# In[236]:


# # Aplicando a funcao para os dados de treino e teste

# # cut
# mean_sort_cut, train = categ_feature('cut', train)
# test['cut'] = test['cut'].replace(mean_sort_cut, inplace = False)

# #color
# mean_sort_color, train = categ_feature('color', train)
# test['color'] = test['color'].replace(mean_sort_color, inplace = False)

# #clarity
# mean_sort_clarity, train = categ_feature('clarity', train)
# test['clarity'] = test['clarity'].replace(mean_sort_clarity, inplace = False)

# # Visualizando teste para checar funcionamento
# test.head()


# In[237]:


# Criando a matriz de correlacao novamente para analise
corr_matrix = train.corr()

plt.subplots(figsize = (10, 10))
sns.heatmap(corr_matrix, square=True, cbar=True, annot = True, cmap='Spectral')
plt.show()


# A partir da matriz de correlação, é possível observar que os dados transformados não possuem correlação alta com o preço. Entretanto, dependendo do modelo a ser aplicado, eles podem apresentar influências cruciais, provocando melhores métricas de ajuste fino do modelo, por exemplo.

# # Treino e Análise de Resultados

# ## Preparação

# Após aplicar todos os tratamentos e entender como cada atributo interage com o alvo, é necessário dividir o conjunto de treino em dois, um de treino e um de pseudo-teste, a fim de avaliar o comportamento de cada modelo para verificar qual é a melhor escolha para a aplicação em questão.  
# É importante ressaltar que, para realizar esta separação, o conjunto de treino deve ser previamente separado em duas partes específicas: x, com todos os atributos, e y, com o alvo, como pode ser observado abaixo.

# In[238]:


# Criando o conjunto de treino e de teste para treinar o modelo a partir de train modificado
x = train.drop(['price'], axis = 1)
y = train['price']

# Criacao de conjuntos de treino e pseudo-teste a partir do conjunto geral de treino
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 2, test_size=0.3)


# Cria-se, adicionalmente, um dicionário de comparação que irá armazenar todos os RMSPE analisados durante os próximos passos. Este dicionário servirá para demonstrar o modelo que gerou o melhor resultado.  
# 
# Além dele, criam-se também duas funções, uma responsável pelo cálculo do RMSPE, já que esta métrica não estava disponível nos modelos utilizados, e uma responsável pela análise de cada modelo, capaz de gerar o treinamento, predição e calcular as métricas de cada caso.

# In[239]:


# Arrays de referência para comparação entre modelos
model_dict = {'Linear Regressor': 1, 'Lasso Regression': 1, 'Ridge Regression': 1, 'AdaBoost Regression': 1,             'Gradient Boosting Regression': 1, 'Random Forest Regression': 1, 'Extra Trees Regression': 1}


# In[240]:


# Função que calcula o RMSPE para validacao dos modelos
def rmspe_score(y_test, y_pred):
    
    rmspe = np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis = 0))

    return rmspe


# In[241]:


# Funcao de regressao generica, para varios modelos diferentes
def model_analysis(X_train, X_test, y_train, y_test, regressor, name):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('')
    print('###### {} ######'.format(name))
    print('Score : %.6f' % regressor.score(X_test, y_test))

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


# ## Modelos Testados

# A partir da preparação do ambiente, aplicam-se os mesmos dados para todos os regressores abaixo, comparando os scores e todas as métricas de erros. A métrica mais importante para este caso é o RMSPE, já que o mesmo é o que será considerado como avaliador durante a competição.

# ### Linear Regression

# In[242]:


get_ipython().run_cell_magic('time', '', "\nlr = LinearRegression()\nmodel_analysis(X_train, X_test, y_train, y_test, lr, 'Linear Regressor')")


# ### Lasso Regression

# In[243]:


get_ipython().run_cell_magic('time', '', "\nlar = Lasso(normalize = True)\nmodel_analysis(X_train, X_test, y_train, y_test, lar, 'Lasso Regression')")


# ### Ridge Regression

# In[244]:


get_ipython().run_cell_magic('time', '', "\nrr = Ridge(normalize = True)\nmodel_analysis(X_train, X_test, y_train, y_test, rr, 'Ridge Regression')")


# ### AdaBoost Regression

# In[245]:


get_ipython().run_cell_magic('time', '', "\nabr = AdaBoostRegressor(random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, abr, 'AdaBoost Regression')")


# ### Gradiente Boosting Regression

# In[246]:


get_ipython().run_cell_magic('time', '', "\ngbr = GradientBoostingRegressor(n_estimators = 200, min_samples_leaf = 2, min_samples_split = 5, \\\n                                max_depth = 10, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, gbr, 'Gradient Boosting Regression')")


# ### Random Forest Regression

# In[247]:


get_ipython().run_cell_magic('time', '', "\nrfr = RandomForestRegressor(n_estimators = 250, n_jobs = 2, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, rfr, 'Random Forest Regression')")


# ### Extra Trees Regression

# In[248]:


get_ipython().run_cell_magic('time', '', "\netr = ExtraTreesRegressor(n_estimators = 1000, n_jobs = -1, random_state = 2)\nmodel_analysis(X_train, X_test, y_train, y_test, etr, 'Extra Trees Regression')")


# ### Comparação

# In[249]:


compare = pd.DataFrame()
compare['Model'] = model_dict.keys()
compare['RMSPE'] = model_dict.values()

compare = compare.set_index('Model').sort_values(['RMSPE'])
compare


# Como o modelo que apresentou menor RMSPE foi o *Gradient Boosting Regressor*, o mesmo foi selecionado para sofrer otimizações de parâmetros e ser retestado a cada nova descoberta, como pode ser observado nos passos a seguir.

# ## Otimização de Parâmetros

# A partir da escolha do modelo, pesquisas relacionadas e dados empíricos, definiram-se parâmetros iniciais para aplicar dois métodos de otimização e validação cruzada considerando 3 combinações diferentes dos conjuntos de dados:
# 1. Random Search;
# 2. Grid Search.

# ### Random Search

# In[250]:


# Definindo a grid para aplicar RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(500, 2500, 5)]
max_features = [7, 8, 9]
max_depth = [10, 12, 14]
min_samples_split = [2, 3, 4]
min_samples_leaf = [2, 4, 6]
bootstrap = [True]
random_state = [2]
learning_rate = [round(float(x), 3) for x in np.linspace(0.01, 0.15, 15)]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'random_state': random_state,
               'learning_rate': learning_rate}

print(random_grid)


# In[62]:


# Aplicando RandomizedSearchCV ao Gradient Boosting Regressor

# Random search com 100 iterações
gbr = GradientBoostingRegressor()
gbr_rs = RandomizedSearchCV(estimator = gbr, 
                            param_distributions = random_grid, 
                            n_iter = 100, 
                            cv = 3, 
                            verbose = 2, 
                            random_state = 2, 
                            n_jobs = -1,
                            scoring = make_scorer(rmspe_score, greater_is_better = False)
)

# Treina o modelo com 100 possibilidades aleatorias dentro do conjunto
# definido pelo random_grid
gbr_rs.fit(x, y)

# Dentro destes treinamentos, define o que apresentou melhor resultado
gbr_rs.best_params_


# A partir dos parâmetros resultantes do método de Random Search, cria-se uma nova grid em torno destes valores para gerar um novo conjunto de teste. Este conjunto será responsável pela avaliação do método de Grid Search.

# ### Grid Search

# In[56]:


# De acordo com os resultados da RandomizedSearch,
# seto os parametros da param_grid em torno deles
# para descobrir o melhor de todos
x = train.drop(['price'], axis = 1)
y = train['price']

param_grid = {
    'learning_rate': [0.02, 0.04, 0.05],
    'max_depth': [10],
    'max_features': [7, 8, 9],
    'min_samples_leaf': [2],
    'min_samples_split': [5, 6],
    'n_estimators': [300, 500, 700, 1000],
    'random_state': [2]
}

gbr = GradientBoostingRegressor()
gbr_grid = GridSearchCV(estimator = gbr, 
                        param_grid = param_grid, 
                        cv = 3,
                        verbose = 2,
                        n_jobs = -1,
                        scoring = make_scorer(rmspe_score, greater_is_better = False)
)

# Treina o modelo com todas as combinacoes
# do conjunto de param_grid
gbr_grid.fit(x, y)

# Define os melhores parametros pro Gradient
# Boosting Regressor dentro deste conjunto
gbr_grid.best_params_


# ### Verificação dos Parâmetros

# Após a definição dos parâmetros otimizados para o modelo escolhido, é necessário aplicá-los aos conjuntos de treino e pseudo-teste para verificação e, posteriormente, gerar o preditor para o conjunto de teste real.

# Este passo sofreu diversas alterações durante o desenvolvimento do modelo e está de acordo com a última submissão da autora, em que são utilizados os parâmetros resultantes do *Grid Search*, sem tratamento de outliers e com atributos categóricos substituídos por números inteiros variando de 1 ao número de categorias de cada um.

# #### Avaliação do Modelo

# In[255]:


get_ipython().run_cell_magic('time', '', "\ngbr = GradientBoostingRegressor(learning_rate = 0.04, max_depth = 10, max_features = 8, \\\n                                min_samples_leaf = 6, min_samples_split = 2, n_estimators = 500, \\\n                                random_state = 2)\n\nmodel_analysis(X_train, X_test, y_train, y_test, gbr, 'Gradient Boosting Regression')")


# #### Arquivo Submetido

# In[256]:


get_ipython().run_cell_magic('time', '', "x = train.drop(['price'], axis = 1)\ny = train['price']\n\ngbr = GradientBoostingRegressor(learning_rate = 0.04, max_depth = 10, max_features = 8, \\\n                                min_samples_leaf = 6, min_samples_split = 2, n_estimators = 500, \\\n                                random_state = 2)\n\ngbr.fit(x, y)\ny_pred = gbr.predict(test)\n\nsubmission = pd.DataFrame({'id':test.index, 'price':y_pred})\n\nsubmission.to_csv('data/submission.csv', index = False)")


# # Conclusão

# O trabalho foi de extrema importância para o desenvolvimento do conhecimento da autora sobre o assunto e gerou motivação para a realização de um bom modelo. Com o aprendizado obtido a partir dele, foi possível conceber novas ideias de Aprendizado de Máquina e definir métricas de melhoria para trabalhos futuros, onde se verificariam a robustez e capacidade de generalização do modelo de forma ainda mais específica, utilizando métricas de verificação da variação durante a validação cruzada, etc.  
# 
# O resultado obtido pelo código da maneira como ele se encontra, porém, não foi o melhor de todos. Durante alguns dos testes, a autora trocou os parâmetros de *min_samples_leaf* e *min_samples_split* erroneamente e, nestas condições, o modelo apresentou o melhor resultado possível e conseguiu atingir o primeiro lugar, público e privado, na competição (na primeira data de término, às 0h de 13/07). Porém, como ele foi gerado por uma confusão e não estava de acordo com a lógica desenvolvida, preferiu-se removê-lo deste relatório e das submissões consideradas pela avaliação privada.  
# 
# Uma solução possível para incluí-lo na lógica deste documento seria colocar a combinação de parâmetros dentro do conjunto considerado pelo *Grid Search* e rodá-lo novamente para verificar se ele se encaixaria na melhor escolha de parâmetros possível. Entretanto, como não houve tempo hábil para realizar este ajuste antes que a competição acabasse, esta solução será aplicada posteriormente, para melhoria do modelo.
