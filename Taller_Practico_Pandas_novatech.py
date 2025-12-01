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

url = "https://raw.githubusercontent.com/mabetancurs/bootcamp_basico_IA/main/DB_Novatech.csv"
bd_original_novatech = pd.read_csv(url, sep=";")

# =====================================
# 2. Exploración Inicial
# =====================================
#primeras 5 filas
print(bd_original_novatech.head())

#ultimas 5 filas
print(bd_original_novatech.tail())

#dimensiones del data set
print(bd_original_novatech.shape)

#nombre de las columnas
print(bd_original_novatech.columns)

#tipo de datos y valores nulos
print(bd_original_novatech.info())

#estadistica basica
print(bd_original_novatech.describe())

# Número de valores únicos por columna
print(bd_original_novatech.nunique())

# Estadísticas para columnas categóricas (tipo object o string)
print("Estadísticas de variables categóricas:")
print(bd_original_novatech.describe(include='object'))

# Conteo de filas duplicadas
print("Número de filas duplicadas:")
print(bd_original_novatech.duplicated().sum())

# Ver primeras filas duplicadas si existen
print("Filas duplicadas (si existen):")
print(bd_original_novatech[bd_original_novatech.duplicated()].head())

# Si quieres ver la correlación entre variables numéricas:
print("Correlación entre variables numéricas:")
print(bd_original_novatech.corr(numeric_only=True))


# =====================================
# 3. Limpieza de Datos
# =====================================
# valores Nulos


print(bd_original_novatech.isnull().sum())

# Ver porcentaje de nulos por columna
print((bd_original_novatech.isnull().mean() * 100).round(2))


# Eliminamos columnas no útiles para análisis inicial
bd_original_novatech.drop(columns=[
    'Nombre', 'Segundo nombre', 'Primer apellido', 'Segundo apellido', 'Fecha nacimiento', 'No Factura', 'Fecha',
    'Codigo DPTO', 'Dpto', 'Codigo Municipio', 'Mun', 'Tipo entidad', 'Nombre Entidad ', 'Fecha1', 'Fecha2', 'Fecha3', 
    'Ruta', 'Milnombre', 'Tipodiag', 'Medico', 'Especialidad', 'Departamento', 'Municipio'
    ], inplace=True) # Borrado inicial


# Imprimimos nuevamente la BD para verificar
print(bd_original_novatech.columns)


# Imputación de valores nulos en variables seleccionadas

# 'Edad': usamos la mediana porque es robusta ante outliers
# bd_original_novatech['Edad'].fillna(bd_original_novatech['Edad'].median(), inplace=True)  # Puede generar error de compatibilidad

bd_original_novatech['Edad'] = bd_original_novatech['Edad'].fillna(
    bd_original_novatech['Edad'].median()) # No genera error de compatibilidad


# 'Conexion': usamos la moda (valor más frecuente) para conservar la categoría dominante
#bd_original_novatech['Conexion '].fillna(bd_original_novatech['Conexion '].mode()[0], inplace=True)  # Puede generar error de compatibilidad

bd_original_novatech['Conexion'] = bd_original_novatech['Conexion'].fillna(
    bd_original_novatech['Conexion'].mode()[0]) # No genera error de compatibilidad


print(bd_original_novatech.isnull().sum())

#-----Identificar y eliminar valores duplicados en el DATASET------

#True indica filas duplicadas respecto a la primera ocurrencia
print(bd_original_novatech.duplicated())  # Serie booleana mostrando duplicados

#Número total de filas duplicadas
print(bd_original_novatech.duplicated().sum())  # Conteo de filas duplicadas

# Opcional: eliminar duplicados para continuar con un dataset depurado
#bd_original_novatech = bd_original_novatech.drop_duplicates().copy()  # Eliminamos duplicados y copiamos el resultado para evitar vistas

#-----Identificar y eliminar valores duplicados en el DATASET para la columna CEDULA------

#Mostrar cuáles cedulas estan duplicadas
print(bd_original_novatech.duplicated(subset=['Cedula']))

# Contar cuántas cédulas están duplicadas
print(bd_original_novatech.duplicated(subset=['Cedula']).sum())

# Eliminar filas duplicadas basándose únicamente en la cédula
bd_original_novatech = bd_original_novatech.drop_duplicates(subset=['Cedula']).copy()


# =====================================
# 4. Transformación de Variables
# =====================================

##Renombrar una Columna

# bd_original_novatech.rename (columns={"Conexion ": "Conexion"}, inplace=True) # se cambia 'Conexion ' por 'Conexion'
# print(bd_original_novatech.columns) # No se uso

#nombre de las columnas
print(bd_original_novatech.columns)# No se uso

##Reordenar una columna : cambia el orden de las columnas

bd_reordenada_novatech = bd_original_novatech[[
    'Cedula', 'Sexo', 'Edad', 'Departamento codigo', 'Municipio codigo', 'Codigo Entidad', 'Conexion', 'Conexion '
    ]]
print(bd_reordenada_novatech.columns)

#es una función anónima que devuelve: 1 si la edad es mayor o igual a 60 años, 0 en caso contrario.

#bd_reordenada_novatech['Adulto mayor'] = bd_reordenada_novatech['Edad'].apply(lambda x: 1 if x >= 60 else 0)# Puede generar Warning sino es un DATAFRAME real

bd_reordenada_novatech.loc[:, 'Adulto mayor'] = bd_reordenada_novatech['Edad'].apply(
    lambda x: 1 if x >= 60 else 0
) #En caso de warning usar esta funcion, con .loc pandas sabe que quieres modificar el DataFrame original sin ambigüedad. 

#Luego de crear la columna Adulto mayor, reordeno nuevamente el DATASET

bd_reordenada_novatech = bd_reordenada_novatech[[
    'Cedula', 'Sexo', 'Edad', 'Adulto mayor', 'Departamento codigo', 'Municipio codigo', 'Codigo Entidad',
    'Conexion', 'Conexion '
    ]]

print(bd_reordenada_novatech.columns)



# =====================================
# 5. Análisis Univariado
# =====================================

# Se procede a realizar un analisis univariado con las siguientes vairables:

#     Departamento
#     Código departamento
#     Municipio
#     Código municipio
#     Edad
#     Adulto mayor (0/1)
#     Sexo
#     Conexión (0 = no tiene / 1 = sí tiene)

# EDAD:

# Qué analizar
    # Distribución de edades (histograma).
    # Promedio, mediana, percentiles.

# Para qué sirve
    # Ver si tu base tiene una población joven, adulta o muy envejecida.
    # Identificar si hay concentración alta de adultos mayores.

bd_reordenada_novatech['Edad'].describe()
bd_reordenada_novatech['Edad'].hist(bins=38, color="blue", edgecolor="black", alpha=0.7)

# ADULTO MAYOR (0/1)

# Qué analizar
    # Cuántos adultos mayores hay en total.
    # Porcentaje de adultos mayores.

# Para qué sirve
    # Es clave para EPS: muestra el tamaño del grupo vulnerable.

bd_reordenada_novatech['Adulto mayor'].value_counts(normalize=True)*100
bd_reordenada_novatech['Adulto mayor'].value_counts().plot(kind='bar')

# SEXO:

# Qué analizar
    # Distribución hombre / mujer.

# Para qué sirve
    # No es un predictor central para conexión, pero sí da perfil demográfico útil.

bd_reordenada_novatech['Sexo'].value_counts(normalize=True)*100
bd_reordenada_novatech['Sexo'].value_counts().plot(kind='bar', color="red")

# CONEXION (0/1)

# Qué analizar
    # Porcentaje de personas sin conexión → este es el dato clave.
    # Distribución por municipio y luego por departamento.

# Para qué sirve
    # Es la justificación para EPS:
        # "X% de la población no tiene conexión y tiene riesgo de perder sus citas."

bd_reordenada_novatech['Conexion'].value_counts(normalize=True)*100
bd_reordenada_novatech['Conexion'].value_counts().plot(kind='bar', color="green")

# MUNICIPIO

# Qué analizar
    # Municipios con mayor concentración poblacional.
    # Municipios donde luego cruzarás “adulto mayor SIN conexión”.
    
bd_reordenada_novatech['Municipio codigo'].value_counts()


# DEPARTAMENTO

# Qué analizar
    # Cuántos registros por departamento.  
    # Departamentos con más o menos riesgo (mirando conexión después).

bd_reordenada_novatech['Departamento codigo'].value_counts()


# =====================================
# 6. Análisis Bivariado
# =====================================

# Mi variable clave es conexión (0/1) donde 0 no tiene conexión y riesgo de perder la cita y 1 es si tiene conexión. 
# Esta variable es la que voy a comprar en mi análisis bivariado vs otras variables

# Conexión vs. Adulto Mayor

# Qué analiza
    # Te muestra si los adultos mayores tienen más riesgo de no tener conexión.

# Por qué importa
    # Es el análisis más fuerte para EPS:
        # "Los adultos mayores son el grupo más afectado por falta de conexión."

pd.crosstab(bd_reordenada_novatech['Adulto mayor'], bd_reordenada_novatech['Conexion'], normalize='index') * 100

(
    pd.crosstab(
        bd_reordenada_novatech['Adulto mayor'],
        bd_reordenada_novatech['Conexion '],
        normalize='index'
    ) * 100
).plot(kind='bar', stacked=True)


# Conexión vs. Edad

# Qué analiza
    # Relación entre edad y falta de conexión.

# Por qué importa
    # Puedes mostrar que a mayor edad -> menor acceso digital.

bd_reordenada_novatech['Rango_Edad'] = pd.cut(
    bd_reordenada_novatech['Edad'],
    bins=[0, 29, 44, 59, 74, 120],
    labels=['18-29', '30-44', '45-59', '60-74', '75+']
)

pd.crosstab(bd_reordenada_novatech['Rango_Edad'], bd_reordenada_novatech['Conexion '], normalize='index') * 100


# Conexión vs. Municipio

# Qué analiza
#     Porcentaje de personas sin conexión por municipio.

# Por qué importa
#     Este es el resultado estrella para presentar a EPS.
#     EPS gestionan por sedes territoriales, por tal razon el municipio es crucial.

bd_reordenada_novatech['Conexion '] = bd_reordenada_novatech['Conexion '].map({
    'SI': 1,
    'NO': 0
}) #Con esta funcion tuve que pasar los datos en cadeta de 'SI' y 'NO' de la columna 'Conexion ' a numeros


conexion_mun = bd_reordenada_novatech.groupby('Municipio')['Conexion '].mean() * 100 # % de personas con conexion
conexion_mun.sort_values()  # del peor al mejor

sin_conexion_mun = (1 - bd_reordenada_novatech.groupby('Municipio')['Conexion '].mean()) * 100 # % de personas sin conexion
sin_conexion_mun.sort_values()  # del peor al mejor


# Conexión vs. Departamento

# Qué analiza
#     Lo mismo que municipio, pero resumido por departamento.

conexion_dep = bd_reordenada_novatech.groupby('Departamento')['Conexion '].mean() * 100 # % de personas con conexion
conexion_dep.sort_values()  # del peor al mejor

sin_conexion_dep = (1 - bd_reordenada_novatech.groupby('Departamento')['Conexion '].mean()) * 100
sin_conexion_dep.sort_values()



# Adulto Mayor SIN conexión por municipio

riesgo_mun = bd_reordenada_novatech[bd_reordenada_novatech['Adulto mayor'] == 1].groupby('Municipio')['Conexion '].mean()
riesgo_mun = (1 - riesgo_mun) * 100  # % sin conexión
riesgo_mun.sort_values()


# Adulto Mayor SIN conexión por departamento

riesgo_dep = bd_reordenada_novatech[bd_reordenada_novatech['Adulto mayor'] == 1].groupby('Departamento')['Conexion '].mean()
riesgo_dep = (1 - riesgo_dep) * 100
riesgo_dep.sort_values()

"----------------------------------------------------------------------------------------------------------------------------"
# Municipio vs. Adulto Mayor

# Qué analiza
    # Municipios con mayor presencia de adultos mayores.
    
adulto_mun = bd_reordenada_novatech.groupby('Municipio')['Adulto mayor'].mean() * 100


# Adulto Mayor vs. Departamento

# Qué analiza
    # Qué departamentos tienen mayor proporción de adultos mayores (grupo vulnerable).

adulto_dep = bd_reordenada_novatech.groupby('Departamento')['Adulto mayor'].mean() * 100
"----------------------------------------------------------------------------------------------------------------------------"


# =====================================
# 7. Análisis Multivariado
# =====================================

# El objetivo es analizar cómo varias variables juntas influyen en la probabilidad de no tener conexión, que se traduce en
# riesgo de perder citas médicas por falta de conectividad.

# Conexión vs. Edad + Municipio
    # Municipios donde los adultos mayores específicos están más desconectados.

tabla_mun = bd_reordenada_novatech.pivot_table(
    index='Municipio',
    columns='Rango_Edad',
    values='Conexion ',
    aggfunc='mean'
) * 100

# Conexión vs. Departamento + Adulto Mayor
    # Departamentos donde los adultos mayores específicos están más desconectados.

tabla_dep = bd_reordenada_novatech.pivot_table(
    index='Departamento',
    columns='Adulto mayor',
    values='Conexion ',
    aggfunc='mean'
) * 100


# =====================================
# 8. Correlación Numérica
# =====================================

# Matriz de correlación (solo numéricas)

# Aunque Conexión es binaria (0/1), se va a correlacionar con:

    # Edad
    # Adulto_mayor
    # Código departamento
    # Código municipio

#Interpretacion

    # Correlación negativa entre Edad y Conexión = a mayor edad -> menos conexión
    # Alta correlación entre Edad y Adulto Mayor = esperado
    
correlacion_num = bd_reordenada_novatech[['Edad', 'Adulto mayor', 'Conexion ']].corr()

# Esta tabla que mide la relación estadística entre cada par de variables numéricas del dataset.
# Los valores van de:
    # 	+1.0 → correlación positiva perfecta
    # 	0.0 → sin relación
    # 	−1.0 → correlación negativa perfecta

# 1. Correlación Edad ↔ Adulto Mayor: Se espera un valor positivo alto (cerca de 0.7–0.9)
# Interpretación
    # Entre más alta la edad, más probable es que la persona sea "adulto mayor".
    # Eso simplemente confirma que la columna "Adulto Mayor" está bien construida.

# 2. Correlación Edad <==> Conexión: Si el número es negativo Ej: (−0.25, −0.35)
# Entre más edad tiene la persona, menos probabilidad de tener conexión.
# Conclusión: Los adultos mayores son más vulnerables digitalmente.

# 3. Correlación Adulto Mayor <==>  Conexión
# Si la hipótesis es correcta esto también suele ser negativo. Ejemplo: −0.40
# Interpretación
    # Ser adulto mayor está asociado a una menor probabilidad de estar conectado.


# =====================================
# 9. Estadísticas Detalladas
# =====================================

#Edad – Media, Mediana y Desviación por Adulto Mayor
    # Muestra cómo cambia la edad dentro de cada grupo (mayor / no mayor).
    # Útil para verificar si el etiquetado "Adulto Mayor" tiene coherencia.

print(bd_reordenada_novatech.groupby('Adulto mayor')['Edad'].agg(['mean', 'median', 'std']))



#Porcentaje de conexión por sexo y adulto mayor (Pivot Table):
# Promedio de supervivencia por sexo y clase
print(bd_reordenada_novatech.pivot_table(values='Conexion_num', index='Sexo', columns='Adulto mayor', aggfunc='mean'))
print(bd_reordenada_novatech.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean'))
                                                  

#Edad por Departamento (describe completo)
    # Esto revela si hay departamentos más envejecidos que otros.
    # Es clave para segmentar riesgo de desconexión.
print(bd_reordenada_novatech.groupby('Departamento')['Edad'].describe())


#Promedio de edad según conexión
    #Indica si las personas sin conexión son significativamente mayores.
print(bd_reordenada_novatech.groupby('Conexion ')['Edad'].mean())


#Proporción de adultos mayores por municipio
    # Muestra qué municipios están más envejecidos.
    # Más adultos mayores -> más riesgo -> más desconexión.
print(bd_reordenada_novatech.groupby('Municipio')['Adulto mayor'].mean())


# =====================================
# 11. Exportar Dataset Limpio
# =====================================
bd_reordenada_novatech.to_csv("db_novatech_limpia.csv", index=False)

