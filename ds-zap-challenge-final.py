import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Arquivo Treino
# Ler coleção de json | list output
with open('C:/Users/Lucas Reis/Desktop/Lucas/DataScience/Testes/ZAP/source-4-ds-train.json/source-4-ds-train.json', encoding="utf8") as file_data:
    data_3 = file_data.readlines()

# Json.loads() quando temos strings
for i, item in enumerate(data_3):
    data_3[i] = json.loads(item)

# Arquivo Teste
with open('C:/Users/Lucas Reis/Desktop/Lucas/DataScience/Testes/ZAP/source-4-ds-test.json/source-4-ds-test.json', encoding="utf8") as file_data:
    data_4 = file_data.readlines()

# Json.loads() quando temos strings
for i, item in enumerate(data_4):
    data_4[i] = json.loads(item)

''' Como estamos usando dados de imóveis, é visto que o valor pode variar muito da média quando o imóveis está bem localizado e
 tem tamanho acima da média
 Adicionado uma variável k para usaramos como controle de outlier '''


def remove_outlier(dataset, nome_col, k):
    q1 = dataset[nome_col].quantile(0.25)
    q3 = dataset[nome_col].quantile(0.75)
    iqr = q3-q1  # Interquartile range
    qbaixo = q1-k*iqr
    qcima = q3+k*iqr
    dataset_saida = dataset.loc[(dataset[nome_col] > qbaixo) & (dataset[nome_col] < qcima)]
    print(qbaixo)
    print(qcima)
    return dataset_saida


data_3 = json_normalize(data_3)
data_4 = json_normalize(data_4)

data_4 = data_4.drop(['pricingInfos.price', 'pricingInfos.rentalTotalPrice'], axis=1)

for coluna_med in [
                'bedrooms', 'address.geoLocation.location.lat', 'address.geoLocation.location.lon'
                , 'bathrooms', 'suites', 'parkingSpaces', 'pricingInfos.monthlyCondoFee'
                , 'pricingInfos.monthlyCondoFee', 'pricingInfos.yearlyIptu', 'totalAreas', 'usableAreas']:
    data_4[coluna_med] = data_4[coluna_med].fillna(data_4[coluna_med].median())


# Removendo os outliers
k = 10
for coluna_out in [
                    'parkingSpaces', 'usableAreas', 'suites', 'bathrooms', 'bedrooms', 'totalAreas'
                    ,'pricingInfos.price', 'pricingInfos.monthlyCondoFee', 'pricingInfos.yearlyIptu']:
    data_3 = remove_outlier(data_3, coluna_out, k)


# Fill NA com a mediana
data_3['bedrooms'] = data_3['bedrooms'].fillna(data_3['bedrooms'].median())


# Mudança de tipos de dados
for coluna_type in [
                    'pricingInfos.price', 'pricingInfos.yearlyIptu', 'pricingInfos.rentalTotalPrice'
                    , 'pricingInfos.monthlyCondoFee', 'parkingSpaces'
                    , 'suites', 'bathrooms', 'totalAreas', 'bedrooms'
                    , 'address.geoLocation.location.lon', 'address.geoLocation.location.lat']:
    data_3[coluna_type].astype(float)


# Começarei deletando algumas features que na minha opnião, não seram nescessárias ainda
data_3 = data_3.drop([
            'address.city', 'address.country',
            'address.district', 'address.geoLocation.precision', 'address.locationId'
            , 'address.neighborhood', 'address.state', 'address.street'
            , 'address.streetNumber'
            , 'address.unitNumber'
            , 'address.zipCode'
            , 'address.zone'
            , 'createdAt'
            , 'description'
            , 'id'
            , 'images'
            , 'listingStatus'
            , 'owner'
            , 'updatedAt'
            , 'title'], axis=1)

data_3 = data_3.dropna(subset=['address.geoLocation.location.lat'])

data_3 = data_3[~data_3['pricingInfos.period'].str.contains('DAILY', na=False)]
data_3 = data_3[~data_3['pricingInfos.period'].str.contains('YEARLY', na=False)]

data_3_rental = data_3[data_3['pricingInfos.businessType'].apply(lambda x:x in ['RENTAL'])]
data_3_sale = data_3[data_3['pricingInfos.businessType'].apply(lambda x:x in ['SALE'])]

data_3_sale = data_3_sale.drop(['pricingInfos.businessType', 'pricingInfos.period', 'pricingInfos.rentalTotalPrice', 'publicationType', 'publisherId'], axis=1)
data_3_rental = data_3_rental.drop(['pricingInfos.businessType','pricingInfos.period','pricingInfos.monthlyCondoFee','pricingInfos.price','publicationType','publisherId'], axis=1)


# A maioria dos algoritimos de ML não se dão muito bem com texo, então usaremos números para representar atributos de texo
from sklearn import preprocessing

le = LabelEncoder()
le.fit(data_4['unitTypes'])
data_4['unitTypes'] = le.transform(data_4['unitTypes'])

le.fit(data_3_sale['unitTypes'])
data_3_sale['unitTypes'] = le.transform(data_3_sale['unitTypes'])

le.fit(data_3_rental['unitTypes'])
data_3_rental['unitTypes'] = le.transform(data_3_rental['unitTypes'])

# Feature Scaling e modelos.

# Transformando todo nosso data frame usando estatistica em uma matriz numpy
# A padronização de dados é um requisito comum para muitos algorítimos de machine learning usando scikit-learn
# Gaussiano com média 0 e variação unitária


# x_scaled_rental =  preprocessing.scale(data_3_rental)
# x_scaled_sale =  preprocessing.scale(data_3_sale)


# train_set, test_set = train_test_split(data_3_sale, test_size=0.2, random_state=42)
# print(len(train_set), "train + ", len(test_set), "test")


x_treino_sale = data_3_sale[[
                            'address.geoLocation.location.lat', 'address.geoLocation.location.lon','bathrooms'
                            , 'bedrooms', 'parkingSpaces', 'pricingInfos.monthlyCondoFee'
                            , 'pricingInfos.yearlyIptu', 'suites', 'unitTypes', 'usableAreas']]
y_treino_sale = data_3_sale['pricingInfos.price']


x_teste_sale = data_4[[
                        'address.geoLocation.location.lat', 'address.geoLocation.location.lon', 'bathrooms'
                        , 'bedrooms', 'parkingSpaces', 'pricingInfos.monthlyCondoFee'
                        , 'pricingInfos.yearlyIptu', 'suites', 'unitTypes', 'usableAreas''']]


x_treino_rent = data_3_rental[[
                                'address.geoLocation.location.lat', 'address.geoLocation.location.lon', 'bathrooms'
                                , 'bedrooms', 'parkingSpaces', 'pricingInfos.yearlyIptu', 'suites'
                                , 'unitTypes', 'usableAreas']]

y_treino_rent = data_3_rental['pricingInfos.rentalTotalPrice']

x_teste_rent = data_4[[
                        'address.geoLocation.location.lat', 'address.geoLocation.location.lon', 'bathrooms'
                        , 'bedrooms', 'parkingSpaces', 'pricingInfos.yearlyIptu', 'suites', 'unitTypes', 'usableAreas']]


# from sklearn.model_selection import cross_val_score


rf = RandomForestRegressor(
                            bootstrap=True, criterion='mse', max_depth=None,
                            max_features=9, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=50,
                            n_jobs=-1, oob_score=False, random_state=None,
                            verbose=0, warm_start=False
                           )


rf.fit(x_treino_sale, y_treino_sale)
y_pred = rf.predict(x_teste_sale)

rf.fit(x_treino_rent, y_treino_rent)
y_pred_2 = rf.predict(x_teste_rent)

final_csv = pd.DataFrame({
                        "Id": data_4["id"],
                        "pricingInfos.price": y_pred_2}
                        )

final_csv.to_csv(r'C:\Users\Lucas Reis\Desktop\Lucas\DataScience\Zap desafio final sem aluguel.csv', index=False)
