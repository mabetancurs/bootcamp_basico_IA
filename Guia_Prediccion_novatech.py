# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 00:20:11 2025

@author: camil
"""

# ============================================================
#  GUÍA INTERACTIVA COMPLETA – CICLO DE VIDA ML CON TITANIC
#  Autor: Diana Carolina
#
#  Objetivo:
#   - Validar que la limpieza de la BD quedó bien
#   - Preparar los datos para ML (preprocesamiento)
#   - Entrenar varios modelos supervisados
#   - Probar un modelo no supervisado
#   - Evaluar, comparar y elegir el mejor modelo
#   - Construir un Pipeline con GridSearchCV
#   - Realizar predicciones y exportar resultados
# ============================================================


# ==========
# 1. IMPORTAR LIBRERÍAS
# ==========

# Pandas: manejo de tablas de datos (DataFrame)
import pandas as pd

# NumPy: operaciones numéricas (lo usaremos para algunas funciones)
import numpy as np

# Matplotlib y Seaborn: visualización básica (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Importas la clase OneHotEncoder, que sirve para convertir texto en números mediante One Hot Encoding.
from sklearn.preprocessing import OneHotEncoder

# StandardScaler: escalamiento de variables numéricas
from sklearn.preprocessing import StandardScaler

# train_test_split: separar datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Opcional: un modelo simple para demostrar uso (ej: Regresión Logística)
from sklearn.linear_model import LogisticRegression

import category_encoders as ce

# ===== IMPORTES ADICIONALES PARA MODELOS Y PIPELINES =====

# Modelos supervisados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modelo no supervisado
from sklearn.cluster import KMeans

# Métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Herramientas avanzadas
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Guardar pipeline entrenado
#PIPELINES permite encadenar todos los pasos de tu proceso de Machine Learning en un solo flujo ordenado, desde el preprocesamiento hasta el modelo final.

import joblib

# Configuración estética opcional para los gráficos
plt.style.use("ggplot")

# Fijar semilla para reproducibilidad
np.random.seed(42)


# ==========
# 2. CARGA DE DATOS
# ==========

# NOTA:
# - Asegúrate de tener un archivo CSV en la misma carpeta del .py
# - Ejemplo: "titanic.csv", "diabetes.csv", "clientes.csv", etc.
# - Cambia el nombre "datos_ml.csv" por el tuyo.

# Ruta del archivo de datos (MODIFICA ESTO)
RUTA_ARCHIVO = "titanic_limpio_ml.csv"

# Cargar el archivo CSV en un DataFrame de Pandas
# sep="," indica que el separador es la coma; puedes cambiarlo a ";"
df1 = pd.read_csv(RUTA_ARCHIVO, sep=",")


# ==========
# 3. COPIA DE DATOS
# ==========
df = df1.copy()


# ==========
# 4. INSPECCIÓN INICIAL DEL DATASET
# ==========

# Mostrar las primeras filas para verificar la carga
print("\n===== 2. PRIMERAS FILAS DEL DATASET =====")
print(df.head())

print("\n===== 3. INFORMACIÓN GENERAL DEL DATASET =====")
# Info general: tipos de datos, cantidad de nulos por columna, etc.
df.info()# Si tengo 10nde tipo objeto 10 tengo que cambiar

print("\n===== 3.1 DIMENSIONES DEL DATASET (FILAS, COLUMNAS) =====")
print("Shape:", df.shape)


# ==========
# 5. REVISION DE VALORES NULOS Y DUPLICADOS
# ==========

print("\n===== 4. VALORES NULOS POR COLUMNA =====")
# isnull() devuelve True/False; sum() cuenta los True
print(df.isnull().sum())

print("\n===== 4.1 CANTIDAD DE FILAS DUPLICADAS =====")
print(df.duplicated().sum())


# ==========
# 6. REVISION DE TIPOS DE DATOS
# ==========

print("\n===== 7. TIPOS DE DATOS ANTES DE CONVERTIR =====")
print(df.dtypes)


# ============================================================
# 7. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================

print("\n===== 8. CODIFICACIÓN DE VARIABLES CATEGÓRICAS =====")

# Detectar columnas categóricas automáticamente
columnas_categoricas = df.select_dtypes(include=["object", "category"]).columns
print("Columnas categóricas detectadas:", list(columnas_categoricas))

# Hacemos una copia para trabajar en la codificación
df2 = df.copy()

#################################################################
# | Variable     | Tipo   | Observaciones                      |
# | ------------ | ------ | ---------------------------------- |
# | embarked     | object | Categórica nominal (c, s, q)       |
# | sex          | object | Categórica binaria (male, female)  |
#################################################################

# --------------------------------------------------------------
# 7.1 Codificación binaria para 'sex'
# --------------------------------------------------------------
# Objetivo:
#   - Convertir 'sex' (male/female) en 0/1
#   - Es una variable binaria → Label/Binary Encoding es suficiente
#   - Ahorra columnas y sirve para todos los modelos

print("\n--- 7.1 Codificación binaria para 'sex' (male=0, female=1) ---")

if "sex" in df2.columns:
    print("Valores únicos de 'sex' antes:", df2["sex"].unique())
    df2["sex"] = df2["sex"].map({"male": 0, "female": 1})
    print("Valores únicos de 'sex' después:", df2["sex"].unique())
else:
    print("⚠ La columna 'sex' no existe. Ajusta el código si tu dataset es diferente.")


# --------------------------------------------------------------
# 7.2 One-Hot Encoding para 'embarked'
# --------------------------------------------------------------
# Objetivo:
#   - Convertir 'embarked' (c, s, q) en variables dummies
#   - Usar One-Hot Encoding (OHE) porque:
#       * Es nominal (no hay orden c > s > q)
#       * Evitamos inventar un orden con Label Encoding
#   - drop_first=True para evitar multicolinealidad en modelos lineales

print("\n--- 7.2 One-Hot Encoding para 'embarked' con get_dummies ---")

if "embarked" in df2.columns:
    print("Valores únicos de 'embarked' antes:", df2["embarked"].unique())
    df2 = pd.get_dummies(df2, columns=["embarked"], prefix="embarked", drop_first=True)
    print("Columnas de 'embarked' después del OHE:")
    print([col for col in df2.columns if col.startswith("embarked_")])
else:
    print("⚠ La columna 'embarked' no existe. Ajusta el código si tu dataset es diferente.")


# --------------------------------------------------------------
# 7.3 (Opcional) Target Encoding para 'embarked' (Sólo demostración) -- NO USAR ES PARA MODELOS NOS SUPERVI
# --------------------------------------------------------------
# Aquí solo mostramos el uso de TargetEncoder como concepto.
# No lo usamos en el modelo final para evitar mezclar enfoques.

print("\n--- 7.3 (OPCIONAL) Target Encoding para 'embarked' (DEMO) ---")

if ("embarked" in df.columns) and ("survived" in df.columns):
    encoder_te = ce.TargetEncoder(cols=["embarked"])
    df_te = df.copy()
    df_te["embarked_te"] = encoder_te.fit_transform(df_te["embarked"], df_te["survived"])
    print("Ejemplo de Target Encoding (primeras filas):")
    print(df_te[["embarked", "embarked_te"]].head())
else:
    print("No se puede demostrar Target Encoding: falta 'embarked' o 'survived'.")


# --------------------------------------------------------------
# 7.4 Resultado final tras codificación
# --------------------------------------------------------------

print("\n===== 8.1 DATASET TRAS CODIFICAR CATEGÓRICAS (df2) =====")
print(df2.head())
print(df2.dtypes)
print("Shape de df2:", df2.shape)


# ============================================================
# 8. PREPARAR DATASET PARA MODELADO (df_modelo)
# ============================================================

print("\n===== 9. PREPARACIÓN DE df_modelo PARA ML =====")

# En este dataset ya NO hay columnas tipo object (sex=0/1, embarked_OHE)
df_modelo = df2.copy()

# Confirmar que no queden columnas object
columnas_object = df_modelo.select_dtypes(include=["object"]).columns
if len(columnas_object) > 0:
    print("⚠ Aún hay columnas object. Se eliminarán:", list(columnas_object))
    df_modelo = df_modelo.drop(columns=columnas_object)
else:
    print("✅ No hay columnas object. Todas son numéricas o categóricas codificadas.")

print("\nColumnas finales en df_modelo:")
print(df_modelo.columns)


# ============================================================
# 9. SELECCIÓN DE VARIABLES (X, y)
# ============================================================

print("\n===== 10. SEPARACIÓN DE X (FEATURES) E y (TARGET) =====")

# Para Titanic, la variable objetivo es 'survived'
NOMBRE_TARGET = "survived"

if NOMBRE_TARGET not in df_modelo.columns:
    raise ValueError(
        f"ERROR: La columna objetivo '{NOMBRE_TARGET}' no existe en df_modelo.\n"
        f"Columnas disponibles: {list(df_modelo.columns)}"
    )

# y: lo que queremos predecir (sobrevivió o no)
y = df_modelo[NOMBRE_TARGET]

# X: todas las demás columnas (features)
X = df_modelo.drop(NOMBRE_TARGET, axis=1)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

print("\nDistribución de la variable objetivo (y):")
print(y.value_counts(normalize=True))


# ============================================================
# 10. ESCALAMIENTO DE VARIABLES NUMÉRICAS
# ============================================================

print("\n===== 11. ESCALAMIENTO DE VARIABLES NUMÉRICAS =====")

# Todas las columnas de X son numéricas (int/float)
columnas_num_X = X.columns
print("Columnas numéricas a escalar:", list(columnas_num_X))

# Creamos el StandardScaler
escalador = StandardScaler()

# Ajuste y transformación
X_escalado = X.copy()
X_escalado[columnas_num_X] = escalador.fit_transform(X[columnas_num_X])

print("\nPrimeras filas de X_escalado:")
print(X_escalado.head())


# ============================================================
# 11. DIVISIÓN EN TRAIN Y TEST
# ============================================================

print("\n===== 12. DIVISIÓN EN TRAIN / TEST =====")

X_train, X_test, y_train, y_test = train_test_split(
    X_escalado,
    y,
    test_size=0.2,        # 20% test
    random_state=42,
    stratify=y            # mantener proporción de clases
)

print("Shape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test :", y_test.shape)


# ============================================================
# 12. ENTRENAMIENTO DE VARIOS MODELOS SUPERVISADOS
# ============================================================

print("\n===== 13. MODELOS SUPERVISADOS (CLASIFICACIÓN) =====")
print("Todos estos modelos son SUPERVISADOS: usan X (features) e y (target).")
print("Problema: clasificación binaria (0 = no sobrevivió, 1 = sobrevivió).")

# Diccionario de modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados_modelos = []

for nombre, modelo in modelos.items():
    print("\n--------------------------------------------------------")
    print(f"Entrenando modelo supervisado: {nombre}")

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    y_pred = modelo.predict(X_test)

    # Métrica principal: Accuracy
    acc = accuracy_score(y_test, y_pred)
    resultados_modelos.append((nombre, acc))

    print(f"Accuracy en test: {acc:.4f}")

    # Matriz de confusión
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Reporte de clasificación (precision, recall, f1-score)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

# Resumen comparativo
print("\n===== 13.1 RESUMEN DE ACCURACY DE CADA MODELO =====")
for nombre, acc in resultados_modelos:
    print(f"{nombre:20s} -> Accuracy = {acc:.4f}")

# Elegir el mejor modelo según Accuracy
mejor_modelo_nombre, mejor_acc = max(resultados_modelos, key=lambda t: t[1])
print(f"\n✅ Mejor modelo (según Accuracy): {mejor_modelo_nombre} con {mejor_acc:.4f}")


# ============================================================
# 13. MODELO NO SUPERVISADO: KMEANS (CLUSTERING)
# ============================================================

print("\n===== 14. MODELO NO SUPERVISADO: KMEANS =====")
print("KMeans es NO SUPERVISADO: sólo usa X (no usa y).")

# Entrenar KMeans con 2 clusters (podrían representar 2 grupos de pasajeros)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_escalado)

labels_kmeans = kmeans.labels_

print("\nTamaño de cada cluster:")
valores, cuentas = np.unique(labels_kmeans, return_counts=True)
for cl, c in zip(valores, cuentas):
    print(f"Cluster {cl}: {c} pasajeros")

# Comparar clusters con la variable real survived (solo para análisis)
print("\nTabla cruzada: Cluster vs Survived")
print(pd.crosstab(labels_kmeans, y))


# ============================================================
# 14. PIPELINE + GRIDSEARCHCV (FLUJO COMPLETO)
# ============================================================

print("\n===== 15. PIPELINE + GRIDSEARCHCV (KNN) =====")
print("Ahora usamos el df ORIGINAL (con 'sex' y 'embarked' como texto) y un Pipeline completo.")

# X_raw: dataset original sin la columna objetivo
X_raw = df.drop("survived", axis=1)
y_raw = df["survived"]

# Definimos qué columnas son numéricas y cuáles categóricas en el df original
columnas_numericas = ["age", "ischild", "parch", "pclass", "sibsp", "tarifa"]
columnas_categoricas_pipe = ["sex", "embarked"]

# Preprocesador: ColumnTransformer aplica transformaciones según el tipo de columna
preprocesador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), columnas_numericas),               # Escala numéricas
        ("cat", OneHotEncoder(drop="first"), columnas_categoricas_pipe)  # OHE en categóricas
    ]
)

# Definimos el Pipeline: preprocesamiento + modelo KNN
pipeline_knn = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("knn", KNeighborsClassifier())
])

# Dividimos train/test para este flujo
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    random_state=42,
    stratify=y_raw
)

# Definimos la grilla de hiperparámetros para GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"]
}

# GridSearchCV: búsqueda exhaustiva de mejores hiperparámetros con validación cruzada
grid_knn = GridSearchCV(
    estimator=pipeline_knn,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nEntrenando GridSearchCV con Pipeline (esto puede tardar un poco)...")
grid_knn.fit(X_train_pipe, y_train_pipe)

print("\nMejores hiperparámetros encontrados:")
print(grid_knn.best_params_)

print("\nMejor Accuracy promedio en validación cruzada:")
print(grid_knn.best_score_)

# Extraer el mejor pipeline ya entrenado
mejor_pipeline_knn = grid_knn.best_estimator_

# Evaluar en el conjunto de prueba
y_pred_pipe = mejor_pipeline_knn.predict(X_test_pipe)

print("\nDesempeño del MEJOR PIPELINE KNN en test:")
print("Accuracy:", accuracy_score(y_test_pipe, y_pred_pipe))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_pipe, y_pred_pipe))
print("\nReporte de clasificación:")
print(classification_report(y_test_pipe, y_pred_pipe))

# Guardar el pipeline a disco
NOMBRE_PIPELINE = "pipeline_titanic_knn.pkl"
joblib.dump(mejor_pipeline_knn, NOMBRE_PIPELINE)
print(f"\n✅ Pipeline guardado como: {NOMBRE_PIPELINE}")


# ============================================================
# 15. PREDICCIONES FINALES Y EXPORTACIÓN
# ============================================================

print("\n===== 16. PREDICCIONES CON EL MEJOR PIPELINE =====")

# Predicciones finales
y_pred_final = mejor_pipeline_knn.predict(X_test_pipe)

# Probabilidades (si el modelo lo soporta)
if hasattr(mejor_pipeline_knn, "predict_proba"):
    y_proba = mejor_pipeline_knn.predict_proba(X_test_pipe)[:, 1]
else:
    y_proba = None

# Construir DataFrame de resultados
df_resultados = pd.DataFrame({
    "y_real": y_test_pipe,
    "y_pred": y_pred_final
}, index=y_test_pipe.index)

if y_proba is not None:
    df_resultados["prob_supervivencia"] = y_proba

NOMBRE_RESULTADOS = "predicciones_titanic_pipeline_knn.csv"
df_resultados.to_csv(NOMBRE_RESULTADOS, index=True)
print(f"✅ Archivo de predicciones exportado como: {NOMBRE_RESULTADOS}")


# ============================================================
# 16. EXPORTAR DATASET LIMPIO PARA OTROS PROYECTOS
# ============================================================

print("\n===== 17. EXPORTAR DATASET LIMPIO PARA ML =====")

NOMBRE_SALIDA = "dataset_limpio_para_ml.csv"
df_modelo.to_csv(NOMBRE_SALIDA, index=False)
print(f"✅ Dataset limpio exportado como: {NOMBRE_SALIDA}")


# ============================================================
# 17. RESUMEN DEL CICLO DE VIDA DEL PROYECTO ML
# ============================================================

print("\n===== 18. RESUMEN DEL CICLO COMPLETO DE ML CON TITANIC =====")
print("""
1. Validamos la limpieza del dataset:
   - Revisamos nulos, duplicados, tipos de datos y dimensiones.
2. Codificamos variables categóricas:
   - 'sex' → 0/1 (male/female).
   - 'embarked' → One-Hot Encoding (columnas embarked_*).
3. Construimos df_modelo sólo con variables numéricas listas para ML.
4. Separamos X (features) e y (target = survived).
5. Escalamos las variables numéricas con StandardScaler.
6. Dividimos en train y test (80%/20%) con estratificación.
7. Entrenamos varios modelos SUPERVISADOS:
   - Regresión Logística
   - KNN
   - SVM (RBF)
   - Árbol de Decisión
   - Random Forest
   - Gradient Boosting
   y comparamos por Accuracy y métricas de clasificación.
8. Probamos un modelo NO SUPERVISADO:
   - KMeans con 2 clusters, analizando cómo se relacionan con 'survived'.
9. Creamos un PIPELINE COMPLETO con:
   - ColumnTransformer (escalado + OneHotEncoder)
   - KNN
   y usamos GridSearchCV para encontrar los mejores hiperparámetros.
10. Evaluamos el mejor pipeline en el conjunto de prueba.
11. Guardamos:
    - El pipeline entrenado → pipeline_titanic_knn.pkl
    - Las predicciones → predicciones_titanic_pipeline_knn.csv
    - El dataset limpio para ML → dataset_limpio_para_ml.csv
12. Con esto completamos TODO el ciclo de vida de un proyecto de Machine Learning
    aplicado al Titanic usando sklearn.
""")

print("\n🎉 FIN DE LA GUÍA INTERACTIVA COMPLETA CON TITANIC Y SKLEARN 🎉")
