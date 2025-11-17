# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:33:20 2025

@author: Leonardo
"""

#%% Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt

#%% Leitura dos dados
df = pd.read_csv('dados/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.info()
df = df.drop(columns='customerID')

# Analise Exploratoria
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.isnull().sum()
df = df.dropna()

# Graficos
tenure_churn_no = df[df['Churn'] == 'No']['tenure']
tenure_churn_yes = df[df['Churn'] == 'Yes']['tenure']

plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.xlabel('tenure')
plt.ylabel('Number of Customers')
plt.title('Customer Churn Prediction')
plt.legend()
plt.show()

#%% Dummização das variaveis categóricas
col_categoricas = []
for col in df.columns:
    if df[col].dtypes == 'object':
        print(f'{col}: {df[col].unique()}')
        col_categoricas.append(col)
        
col_categoricas.remove('Churn')
        
df = df.replace('No phone service', 'No').replace('No internet service', 'No')

df_dummies = pd.get_dummies(df, columns = col_categoricas, drop_first=True, dtype=int)
df_dummies.loc[df_dummies['Churn'] == 'Yes', 'Churn'] = 1
df_dummies.loc[df_dummies['Churn'] == 'No', 'Churn'] = 0

#%% Normalização das colunas
cols_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_dummies[cols_scale] = scaler.fit_transform(df_dummies[cols_scale])

#%% Separando em dados de treino e teste
df_dummies['Churn'] = df_dummies['Churn'].astype('int')
X = df_dummies.drop(columns='Churn')
y = df_dummies['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Criar Neural Network
import tensorflow as tf
from tensorflow import keras

X_train.shape

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(23,), activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
    ])
    
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)

#%% Predict
yp = model.predict(X_test)
yp[:5]

y_predict=[]
for element in yp:
    if element > 0.5:
        y_predict.append(1)
    else:
        y_predict.append(0)

y_predict[:10]
y_test[:10]

#%% Classification Report
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_predict))

import seaborn as sns
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predict)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()
