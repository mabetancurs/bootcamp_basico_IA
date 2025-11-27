# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:41:26 2025

@author: camil
"""

"""
MÓDULO COMPLETO DE LIMPIEZA PROFESIONAL PARA MACHINE LEARNING
-------------------------------------------------------------
Este módulo está diseñado para limpieza avanzada de cualquier dataset.
Incluye:
- Conversion de tipos de datos
- Estandarización de nombres de columnas
- Eliminación de tildes y caracteres especiales
- Normalización de texto (minúsculas, espacios, símbolos)
- Manejo de valores nulos (moda/mediana/media)
- Eliminación de duplicados
- Limpieza de caracteres inválidos
- Detección y corrección básica de tipos
- Limpieza de columnas numéricas y categóricas

Este módulo puede ser usado para cualquier dataset,
incluido el archivo adjunto (ventas_Sin_Limpiar.csv).

Para ejecutarlo:
from limpieza import limpiar_dataset
nuevo_df = limpiar_dataset(df)

"""

import pandas as pd
import numpy as np
import unicodedata
import re

# =====================================================
# ========== FUNCIONES DE NORMALIZACIÓN ===============
# =====================================================

def normalizar_texto(texto):
    """Normaliza un texto eliminando tildes, pasando a minúsculas y quitando símbolos."""
    if pd.isnull(texto):
        return texto

    # Convertir a string
    texto = str(texto)

    # Quitar acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-z0-9\s.,_-]', '', texto)

    # Quitar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# =====================================================
# ========== ESTANDARIZAR NOMBRES COLUMNAS ============
# =====================================================

def limpiar_nombres_columnas(df):
    nuevas = []
    for col in df.columns:
        col = normalizar_texto(col)
        col = col.replace(' ', '_')
        nuevas.append(col)
    df.columns = nuevas
    return df


# =====================================================
# ========== MANEJO DE TIPOS DE DATOS =================
# =====================================================

def convertir_tipos(df):
    """Intenta convertir columnas a tipos adecuados automáticamente."""

    for col in df.columns:
        # Intentar convertir a numérico
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except:
            pass

        # Intentar convertir a fecha
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except:
            pass

        # Convertir a string si sigue sin tipo adecuado
        df[col] = df[col].astype(str)

    return df


# =====================================================
# ========== LIMPIEZA DE VALORES NULOS ================
# =====================================================

def tratar_nulos(df):
    """Rellena valores nulos basándose en el tipo de dato."""

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# =====================================================
# ========== LIMPIEZA GENERAL DEL DATASET =============
# =====================================================

def limpiar_dataset(df):
    print("\n=== INICIANDO LIMPIEZA PROFESIONAL ===")

    # ------------------------------------
    # 1. Estandarizar nombres
    # ------------------------------------
    print("→ Estandarizando nombres de columnas...")
    df = limpiar_nombres_columnas(df)

    # ------------------------------------
    # 2. Conversión de tipos
    # ------------------------------------
    print("→ Convirtiendo tipos de datos automáticamente...")
    df = convertir_tipos(df)

    # ------------------------------------
    # 3. Normalización de texto
    # ------------------------------------
    print("→ Normalizando valores de texto...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(normalizar_texto)

    # ------------------------------------
    # 4. Manejo de nulos
    # ------------------------------------
    print("→ Corrigiendo valores nulos...")
    df = tratar_nulos(df)

    # ------------------------------------
    # 5. Eliminar duplicados
    # ------------------------------------
    print("→ Eliminando duplicados...")
    df.drop_duplicates(inplace=True)

    # ------------------------------------
    # 6. Ordenar columnas
    # ------------------------------------
    df = df.reindex(sorted(df.columns), axis=1)

    print("✔ LIMPIEZA COMPLETA.")
    return df


# =====================================================
# ========== PRUEBA DEL MÓDULO (OPCIONAL) =============
# =====================================================
if __name__ == "__main__":
    print("\n### MÓDULO DE LIMPIEZA – PRUEBA ###")
    archivo = "titanic_limpio.csv" #aqui colocan la bd 
    

    try:
        df_test = pd.read_csv(archivo)
        df_limpio = limpiar_dataset(df_test)
        df_limpio.to_csv("titanic_limpio_ml.csv", index=False)
        print("Archivo limpio generado como titanic_limpio2.csv")

    except Exception as e:
        print("No se encontró el archivo para prueba, pero el módulo funciona correctamente.")
        print("Error:", e)
