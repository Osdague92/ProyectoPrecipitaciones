# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Carga los datos, probando diferentes codificaciones
try:
    data = pd.read_csv('Precipitaciondt.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('Precipitaciondt.csv', encoding='latin-1')
    except UnicodeDecodeError:
        print("Error: No se pudo decodificar el archivo. Verifica la codificación.")
        exit()

# Vista general
print(data.head())  # Primeras filas
print(data.info())  # Información de las columnas y tipos de datos
print(data.describe())  # Estadísticas descriptivas

# Ver valores nulos en el DataFrame
print(data.isnull().sum())

# Porcentaje de valores faltantes por columna
missing_percent = data.isnull().mean() * 100
print(missing_percent)

# Imputar valores nulos en JUNIO con la media de la columna
data['JUNIO'] = data['JUNIO'].fillna(data['JUNIO'].mean())

# Verificar que no haya nulos en JUNIO
print("\nValores nulos en JUNIO después de la imputación:", data['JUNIO'].isnull().sum())

# Imputar valores nulos en una columna categórica con el valor más frecuente
data['CUENCA'] = data['CUENCA'].fillna(data['CUENCA'].mode()[0])

# Verificar nulos
print("\nValores nulos en CUENCA después de la imputación:", data['CUENCA'].isnull().sum())

# Guardar el DataFrame limpio
data.to_csv("Precipitaciondt_limpio.csv", index=False)
print("\nEl archivo limpio ha sido guardado como 'Precipitaciondt_limpio.csv'")

print(data.dtypes)

"""Esto mostrará todos los valores únicos en la columna, incluyendo posibles problemas como espacios, valores vacíos, o errores tipográficos.

"""

#Ver los valores únicos en la columna CUENCA
print(data['CUENCA'].unique())

#Limpiar la columna
# Quitar espacios al inicio y al final de cada valor
data['CUENCA'] = data['CUENCA'].str.strip()

# Convertir todos los valores a minúsculas o mayúsculas para estandarizar
data['CUENCA'] = data['CUENCA'].str.lower()  # o .str.upper()

"""Asegúrate de que no haya valores nulos en la columna CUENCA"""

# Reemplazar valores nulos con un texto como "desconocido" o eliminarlos
data['CUENCA'] = data['CUENCA'].fillna('desconocido')

# O eliminar las filas con valores nulos en CUENCA
# data = data.dropna(subset=['CUENCA'])

# Seleccionar columnas no numéricas
non_numeric_columns = data.select_dtypes(exclude='number')
print(non_numeric_columns.columns)

# Seleccionar columnas numéricas
numeric_data = data.select_dtypes(include='number')

# Agregar la columna CUENCA para agrupar
numeric_data['CUENCA'] = data['CUENCA']

# Convertir columnas categóricas a variables dummy (One-Hot Encoding) o numericas
data = pd.get_dummies(data, columns=['ESTACIÓN', 'MUNICIPIO', 'CUENCA'])

# Ver las primeras filas de los datos después de la conversión
print(data.head())

print(data.columns)

"""Agrupar los datos de precipitaciones por cuenca
Explicación:
precipitation_columns: Esta lista contiene los nombres de las columnas de precipitaciones, que son las que contienen los datos de lluvia para cada mes.

cuenca_columns: Aquí seleccionamos las columnas generadas por One-Hot Encoding que contienen las cuencas. Estas columnas tienen el prefijo CUENCA_r..

groupby(cuenca_columns): Agrupa los datos según las cuencas. Como las columnas de cuenca tienen valores True o False (o 1 y 0), se puede agrupar fácilmente por estas columnas.

mean(): Calcula la media de las precipitaciones para cada grupo de cuenca.
"""

# Definir las columnas de precipitaciones
precipitation_columns = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']

# Seleccionar las columnas de cuenca (aquellas que contienen 'CUENCA_r.')
cuenca_columns = [col for col in data.columns if 'CUENCA_r.' in col]

# Agrupar por las columnas de cuenca y calcular la media de las precipitaciones
precipitation_by_cuenca = data[cuenca_columns + precipitation_columns].groupby(cuenca_columns).mean()

# Mostrar el resultado
print(precipitation_by_cuenca)

"""Análisis e interpretación:
Filas con combinaciones booleanas: Cada fila parece representar una combinación única de valores booleanos para las columnas de cuencas (True o False). Esto indica si una cuenca está activa para una determinada fila.

Columna de valores numéricos (precipitación): La última columna contiene los valores medios de precipitaciones asociadas a la combinación de cuencas activas.

Qué hacer a continuación
Filtrar las filas relevantes: Dado que tienemos varias combinaciones, probablemente solo necesitemos las filas donde una sola cuenca tiene el valor True. Esto simplifica el análisis y asocia directamente cada cuenca con su promedio de precipitaciones.
"""

# Filtrar filas donde exactamente una cuenca está activa (True)
filtered_data = data[(data[cuenca_columns].sum(axis=1) == 1)]

"""Renombrar las cuencas activas: Una vez filtradas, podrías crear una nueva columna que identifique directamente la cuenca activa para esa fila:"""

# Crear una columna con el nombre de la cuenca activa
filtered_data['CUENCA_ACTIVA'] = filtered_data[cuenca_columns].idxmax(axis=1)

"""Organizar datos para presentación: Ahora puedes agrupar por la nueva columna CUENCA_ACTIVA para obtener las precipitaciones medias para cada cuenca:"""

# Agrupar por la cuenca activa y calcular la media
resumen_cuencas = filtered_data.groupby('CUENCA_ACTIVA')[precipitation_columns].mean()
print(resumen_cuencas)

# Generar un conjunto de datos
X = resumen_cuencas.iloc[:, :-1]  # Características: de ENERO a NOVIEMBRE (todas las columnas excepto DICIEMBRE)
y = resumen_cuencas['DICIEMBRE']  # Objetivo: DICIEMBRE
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo (por ejemplo, regresión lineal)
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)

# Graficar Predicciones vs Valores Reales con una línea de referencia
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Línea de referencia')
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs Valores reales")
plt.legend()
plt.show()

print(data.head())

# Preparar X (meses ENERO a NOVIEMBRE) y Y (DICIEMBRE como objetivo)
precipitation_columns = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO',
                         'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']

# Asegurarnos de que todas las columnas están presentes
resumen_cuencas = resumen_cuencas[precipitation_columns]

# Características y objetivo
X = resumen_cuencas.iloc[:, :-1]  # De ENERO a NOVIEMBRE
Y = resumen_cuencas['DICIEMBRE']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""Pasos para ajustar los hiperparámetros
Utilizando herramientas como GridSearchCV o RandomizedSearchCV de sklearn, puedes ajustar los hiperparámetros.
"""

# Definir el modelo base
base_model = RandomForestRegressor(random_state=42)

# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [50, 100, 200],       # Número de árboles en el bosque
    'max_depth': [None, 10, 20, 30],     # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],     # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],       # Mínimo de muestras en una hoja
    'max_features': ['auto', 'sqrt'],    # Máximo de características consideradas para dividir
}

# Configurar la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Ajustar el modelo a los datos de entrenamiento
grid_search.fit(X_train, Y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print(f"Mejores hiperparámetros: {best_params}")

# Evaluar el mejor modelo en el conjunto de prueba
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
rmse = mean_squared_error(Y_test, predictions, squared=False)
print(f"RMSE con los mejores hiperparámetros: {rmse}")

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Realizar predicciones
predictions = model.predict(X_test)

# Evaluar el modelo
rmse = mean_squared_error(Y_test, predictions, squared=False)
print(f"RMSE del modelo: {rmse}")

# Comparar predicciones con valores reales
comparison = pd.DataFrame({'Real': Y_test, 'Predicción': predictions})
print(comparison.head())

""" predecir la precipitación de diciembre para una cuenca específica o promedio:"""

# Crear un nuevo dato de entrada (modificar con datos reales)
nuevo_dato = np.array([[79.16, 94.68, 130.25, 141.75, 128.13, 59.98, 41.65, 52.91, 79.96, 154.06, 110.23]])
prediccion_diciembre = model.predict(nuevo_dato)
print(f"Predicción de diciembre: {prediccion_diciembre[0]}")

# Calcular el coeficiente de determinación
r2_score = model.score(X_test, Y_test)
print(f"R² del modelo: {r2_score}")

plt.scatter(Y_test, predictions)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs Valores reales")
plt.show()

import joblib
joblib.dump(model, "modelo_precipitaciones.pkl")

modelo_cargado = joblib.load("modelo_precipitaciones.pkl")