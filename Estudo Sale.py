import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize

# Ler coleção de json | list output

with open('C:/Users/Lucas Reis/Desktop/Lucas/DataScience/Testes/ZAP/source-4-ds-train.json/source-4-ds-train.json', encoding="utf8") as file_data:
    data_3 = file_data.readlines()

# Json.loads() quando temos strings
for i, item in enumerate(data_3):
    data_3[i] = json.loads(item)

''' Como estamos usando dados de imóveis, é visto que o valor pode variar muito da média quando o imóveis está bem localizado e
 tem tamanho acima da média
 Adicionado uma variável k para usaramos como controle de outlier '''


def remove_outlier(dataset, nome_col,k):
    q1 = dataset[nome_col].quantile(0.25)
    q3 = dataset[nome_col].quantile(0.75)
    iqr = q3-q1  # Interquartile range
    qbaixo  = q1-k*iqr
    qcima = q3+k*iqr
    dataset_saida = dataset.loc[(dataset[nome_col] > qbaixo) & (dataset[nome_col] < qcima)]
    print(qbaixo)
    print(qcima)
    return dataset_saida

data_3 = json_normalize(data_3)



from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data_3, test_size=0.2, random_state=42)


data_3 = train_set
teste = test_set

teste = pd.DataFrame(teste)
#teste_rental = teste[teste['pricingInfos.businessType'].apply(lambda x:x in ['RENTAL'])]
teste_sale = teste[teste['pricingInfos.businessType'].apply(lambda x:x in ['SALE'])]

teste_sale=teste_sale.drop(['pricingInfos.businessType','pricingInfos.period','pricingInfos.rentalTotalPrice','publicationType','publisherId'], axis=1)
#teste_rental = teste_rental.drop(['pricingInfos.businessType','pricingInfos.period','pricingInfos.monthlyCondoFee','pricingInfos.price','publicationType','publisherId'], axis=1)

teste_sale['bedrooms'] = teste_sale['bedrooms'].fillna(teste_sale['bedrooms'].median())
teste_sale['address.geoLocation.location.lat'] = teste_sale['address.geoLocation.location.lat'].fillna(teste_sale['address.geoLocation.location.lat'].median())
teste_sale['address.geoLocation.location.lon'] = teste_sale['address.geoLocation.location.lon'].fillna(teste_sale['address.geoLocation.location.lon'].median())
teste_sale['bathrooms'] = teste_sale['bathrooms'].fillna(teste_sale['bathrooms'].median())
teste_sale['suites'] = teste_sale['suites'].fillna(teste_sale['suites'].median())
teste_sale['parkingSpaces'] = teste_sale['parkingSpaces'].fillna(teste_sale['parkingSpaces'].median())
teste_sale['pricingInfos.monthlyCondoFee'] = teste_sale['pricingInfos.monthlyCondoFee'].fillna(teste_sale['pricingInfos.monthlyCondoFee'].median())
teste_sale['pricingInfos.yearlyIptu'] = teste_sale['pricingInfos.yearlyIptu'].fillna(teste_sale['pricingInfos.yearlyIptu'].median())
teste_sale['totalAreas'] = teste_sale['totalAreas'].fillna(teste_sale['totalAreas'].median())
teste_sale['usableAreas'] = teste_sale['usableAreas'].fillna(teste_sale['usableAreas'].median())







# Removendo os outliers
k = 10
print(teste_sale.describe())
teste_sale = remove_outlier(teste_sale, 'pricingInfos.price',10 )
data_3 = remove_outlier(data_3, 'parkingSpaces', k) # ok
data_3 = remove_outlier(data_3, 'usableAreas', k)   # ok
data_3 = remove_outlier(data_3, 'suites', k)
data_3 = remove_outlier(data_3, 'bathrooms', k)
data_3 = remove_outlier(data_3, 'bedrooms', k)
data_3 = remove_outlier(data_3, 'totalAreas', k)
data_3 = remove_outlier(data_3, 'pricingInfos.price', k)
data_3 = remove_outlier(data_3, 'pricingInfos.monthlyCondoFee', k)
data_3 = remove_outlier(data_3, 'pricingInfos.yearlyIptu', k)
#data_3 = data_3.dropna(subset=['parkingSpaces', 'bathrooms', 'bedrooms', 'usableAreas', 'suites'])
print(teste_sale.describe())

# Fill NA com a mediana
data_3['bedrooms'] = data_3['bedrooms'].fillna(data_3['bedrooms'].median())

# Mudança de tipos de dados
data_3['pricingInfos.price'].astype(float)
data_3['pricingInfos.yearlyIptu'].astype(float)
data_3['pricingInfos.rentalTotalPrice'].astype(float)
data_3['pricingInfos.monthlyCondoFee'].astype(float)
data_3['parkingSpaces'].astype(float)
data_3['suites'].astype(float)
data_3['bathrooms'].astype(float)
data_3['totalAreas'].astype(float)
data_3['bedrooms'].astype(float)
data_3['address.geoLocation.location.lon'].astype(float)
data_3['address.geoLocation.location.lat'].astype(float)

# Começarei deletando algumas colunas que não usaremos e por fim de melhor processo(falta de memória)
data_3 = data_3.drop([
             'address.city', 'address.country',
             'address.district', 'address.geoLocation.precision', 'address.locationId'
             , 'address.neighborhood', 'address.state', 'address.street'
             , 'address.streetNumber'
             , 'address.unitNumber'
             , 'address.zipCode'
             , 'address.zone'
        ,'createdAt'
        ,'description'
        ,'id'
        ,'images'
        ,'listingStatus'
        ,'owner'
        ,'updatedAt'
        ,'title'],axis = 1)

data_3 = data_3.dropna(subset=['address.geoLocation.location.lat'])

data_3 = data_3[~data_3['pricingInfos.period'].str.contains('DAILY', na = False)]
data_3 = data_3[~data_3['pricingInfos.period'].str.contains('YEARLY', na = False)]

#data_3_rental = data_3[data_3['pricingInfos.businessType'].apply(lambda x:x in ['RENTAL'])]
data_3_sale = data_3[data_3['pricingInfos.businessType'].apply(lambda x:x in ['SALE'])]

data_3_sale=data_3_sale.drop(['pricingInfos.businessType', 'pricingInfos.period', 'pricingInfos.rentalTotalPrice', 'publicationType', 'publisherId'], axis=1)
#data_3_rental = data_3_rental.drop(['pricingInfos.businessType','pricingInfos.period','pricingInfos.monthlyCondoFee','pricingInfos.price','publicationType','publisherId'], axis=1)


# A maioria dos algoritimos de ML não se dão muito bem com texo, então usaremos números para representar atributos de texo
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(teste_sale['unitTypes'])
teste_sale['unitTypes'] = le.transform(teste_sale['unitTypes'])

le.fit(data_3_sale['unitTypes'])
data_3_sale['unitTypes'] = le.transform(data_3_sale['unitTypes'])


# Feature Scaling e modelos.

# Transformando todo nosso data frame usando estatistica em uma matriz numpy
# A padronização de dados é um requisito comum para muitos algorítimos de machine learning usando scikit-learn
# Gaussiano com média 0 e variação unitária


#x_scaled_rental =  preprocessing.scale(data_3_rental)
#x_scaled_sale =  preprocessing.scale(data_3_sale)


#train_set, test_set = train_test_split(data_3_sale, test_size=0.2, random_state=42)
#print(len(train_set), "train + ", len(test_set), "test")


x_treino = data_3_sale[['address.geoLocation.location.lat','address.geoLocation.location.lon','bathrooms'
                         ,'bedrooms','parkingSpaces','pricingInfos.monthlyCondoFee'
                     ,'pricingInfos.yearlyIptu','suites','unitTypes','usableAreas']]
y_treino = data_3_sale['pricingInfos.price']


x_teste = teste_sale[['address.geoLocation.location.lat','address.geoLocation.location.lon','bathrooms'
                         ,'bedrooms','parkingSpaces','pricingInfos.monthlyCondoFee'
                     ,'pricingInfos.yearlyIptu','suites','unitTypes','usableAreas''']]
y_teste = teste_sale['pricingInfos.price']



x_treino_scaled = x_treino #preprocessing.scale(x_treino)
x_teste_scale = x_teste #preprocessing.scale(x_teste)


from sklearn.ensemble import RandomForestRegressor


#from sklearn.model_selection import cross_val_score


rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features=9, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=50,
                      n_jobs=-1, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)



'''
# HyperParameter
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators':[10, 30, 50], 'max_features': [8, 9, 10]},
              {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features': [2, 3, 4]}]

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(x_treino_scaled,y_treino)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
#feature_importances = grid_search.best_estimator_.feature_importances_
#print(feature_importances)
#9 / 5060
'''
name_attribs = list()
# rf = grid_search.best_estimator_
rf.fit(x_treino_scaled, y_treino)
feature_importances = rf.feature_importances_
print(feature_importances)
#forest_scores = cross_val_score(rf, x_treino_scaled, y_treino, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
#forest_rmse_scores = np.sqrt(-forest_scores)
#print('Mean:', forest_rmse_scores.mean())

y_pred = rf.predict(x_teste_scale)
from sklearn.metrics import mean_squared_error

final_mse = mean_squared_error(y_teste, y_pred)
final_rmse = np.sqrt(final_mse)
print(final_rmse)


final_csv = pd.DataFrame({"Valor Real": teste_sale["pricingInfos.price"],
                          "Valor Predict":y_pred })

final_csv.to_csv(r'C:\Users\Lucas Reis\Desktop\Lucas\DataScience\Prever com RNF Teste k=6.csv', index=False)