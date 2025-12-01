# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 11:36:32 2025

@author: camil
"""

import pandas as pd  # Librería principal para análisis de datos

# ===============================================================
# 1. Importación de archivos y fuentes de datos
# ===============================================================

# Lectura de archivos locales (comentados para ejemplo)
# df_csv = pd.read_csv("archivo.csv")  # Leer archivo CSV local
# df_excel = pd.read_excel("archivo.xlsx")  # Leer archivo Excel (requiere openpyxl)
# df_json = pd.read_json("archivo.json")  # Leer archivo JSON
# Lectura desde internet
# Leer CSV desde URL
# df_url = pd.read_csv("https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv")  
# Lectura desde una API (requiere requests)
# import requests
# response = requests.get("https://api.example.com/data.json")
# df_api = pd.DataFrame(response.json())  # Convertir JSON a DataFrame
# =====================================
# 1. Carga del Dataset
# =====================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df4 = pd.read_csv(url)

# =====================================
# 2. Exploración Inicial
# =====================================
#primeras 5 filas
print(df4.head())

#ultimas 5 filas
print(df4.tail())

#dimensiones del data set
print(df4.shape)

#nombre de las columnas
print(df4.columns)

#tipo de datos y valores nulos
print(df4.info())

#estadistica basica
print(df4.describe())

# Número de valores únicos por columna

print(df4.nunique())

# Estadísticas para columnas categóricas (tipo object o string)
print("Estadísticas de variables categóricas:")
print(df4.describe(include='object'))

# Conteo de filas duplicadas
print("Número de filas duplicadas:")
print(df4.duplicated().sum())

# Ver primeras filas duplicadas si existen
print("Filas duplicadas (si existen):")
print(df4[df4.duplicated()].head())

# Si quieres ver la correlación entre variables numéricas:
print("Correlación entre variables numéricas:")
print(df4.corr(numeric_only=True))


# =====================================
# 3. Limpieza de Datos
# =====================================
# valores Nulos


print(df4.isnull().sum())

# Ver porcentaje de nulos por columna
print(df4.isnull().mean() * 100)


# Eliminamos columnas no útiles para análisis inicial
df4.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


print(df4.columns)

# df7=df4.drop(index=[0, 6], inplace=True)

# # Eliminar las filas en las posiciones 0 y 2
# df8 = df4.drop(df4.index[[0, 6]]) #No es necesario es solo un ejemplo

# df4(df4.index[0,4])

# Imputación de valores nulos en variables seleccionadas
# 'Age': usamos la mediana porque es robusta ante outliers
df4['Age'].fillna(df4['Age'].median(), inplace=True)  # Mediana para Edad

# 'Embarked': usamos la moda (valor más frecuente) para conservar la categoría dominante
df4['Embarked'].fillna(df4['Embarked'].mode()[0], inplace=True)  # Moda para Embarque

print(df4.isnull().sum())

##Eliminar valores duplicados
#True indica filas duplicadas respecto a la primera ocurrencia
print(df4.duplicated())  # Serie booleana mostrando duplicados

#Número total de filas duplicadas
print(df4.duplicated().sum())  # Conteo de filas duplicadas

# Opcional: eliminar duplicados para continuar con un dataset depurado
df4 = df4.drop_duplicates().copy()  # Eliminamos duplicados y copiamos el resultado para evitar vistas


# =====================================
# 4. Transformación de Variables
# =====================================

##Renombrar una Columna
df4.rename (columns={"Fare": "Tarifa"}, inplace=True) # se cambia Fare por tarifa
print(df4.columns)

#nombre de las columnas
print(df4.columns)

##REordenar una columna : cambia el orden de las columnas
df5 = df4[["Pclass","Survived","Sex",'SibSp', 'Age', 'Parch', 'Tarifa',
       'Embarked']]
print(df5.columns)

#es una función anónima que devuelve: 1 si la edad es menor a 12 años,
#0 en caso contrario.

df5['IsChild'] = df5['Age'].apply(lambda x: 1 if x < 12 else 0)


# =====================================
# 5. Análisis Univariado
# =====================================

#promedio de la edad por clase

print(df5.groupby("Pclass")["Age"].mean())

# promedio de la edad por embarque

# Estadística descriptiva extendida para tarifa
print(df5['Tarifa'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))  # Incluye percentiles personalizados

# Distribución de embarque (conteo por puerto)
#Conteo de pasajeros por puerto de embarque
print(df5['Embarked'].value_counts())  # Conteo simple por categoría


# =====================================
# 6. Análisis Bivariado
# =====================================

# tasa de supervivencia por sexo
print(df5.groupby("Sex")["Survived"].mean())

#tasa de supervivencia por clase
print(df5.groupby("Pclass")["Survived"].mean())

#tabla cruzada clase vs supervivencia
print(pd.crosstab(df5["Pclass"], df5["Survived"]))

# Tabla cruzada normalizada por fila (proporciones por clase)
#Tabla cruzada Pclass vs Survived (proporciones por clase)
#Las columnas son el estado de supervivencia (Survived: 0 = no sobrevivió, 1 = sobrevivió)
#Las filas son las clases (Pclass: 1, 2, 3)
#normalize='index'
#En lugar de conteos, devuelve proporciones por fila (cada fila suma 1).
#.round(3) Redondea los valores a 3 decimales

print(pd.crosstab(df5["Pclass"], df5["Survived"], normalize='index').round(3))  # Normalización por fila


# =====================================
# 7. Análisis Multivariado
# =====================================
#Edad - media, mediana, y desviación por clase
print(df5.groupby("Pclass")["Age"].agg(["mean","median","std"]))  # Agregaciones múltiples

# Supervivencia por (clase, sexo)
print(df5.groupby(["Pclass", "Sex"])["Survived"].mean().unstack())  # Tabla de doble entrada

# Pivot table multivariable: promedio de Tarifa por (Pclass, Embarked)
print(pd.pivot_table(df5, values='Tarifa', index='Pclass', columns='Embarked', aggfunc='mean').round(2))

# Correlación entre variables numéricas
num_cols = df5.select_dtypes(include=['number']).columns  # Selecciona columnas numéricas
print(df5[num_cols].corr().round(3))  # Correlación de Pearson por defecto


# =====================================
# 8. Correlación Numérica
# =====================================
#Correlación con la variable 'Survived'
print(df5.corr(numeric_only=True)['Survived'].sort_values(ascending=False))

# =====================================
# 9. Estadísticas Detalladas
# =====================================
#Edad - Media, Mediana y Desviación por Clase
print(df5.groupby('Pclass')['Age'].agg(['mean', 'median', 'std']))

#Supervivencia por sexo y clase (pivot):
# Promedio de supervivencia por sexo y clase
print(df5.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean'))
                                                  

#Estadísticas de tarifas pagadas por clase
print(df5.groupby('Pclass')['Tarifa'].describe())

#Promedio de edad según supervivencia
print(df5.groupby('Survived')['Age'].mean())

#Proporción de niños por clase
print(df5.groupby('Pclass')['IsChild'].mean())

# =====================================
# 11. Exportar Dataset Limpio
# =====================================
df5.to_csv("titanic_limpio.csv", index=False)

