# Proyecto: Predicción de Precipitaciones Mensuales

## Descripción
Este proyecto desarrolla un modelo predictivo de machine learning para estimar la precipitación de diciembre usando datos históricos de los 11 meses anteriores.  
Se trabajó con datos agrupados por cuencas y se aplicaron modelos de **Regresión Lineal** y **Random Forest Regressor**, siendo este último optimizado mediante **GridSearchCV**.

## Objetivo
Predecir la precipitación mensual de diciembre para cada cuenca, apoyando la planificación agrícola, la gestión de recursos hídricos y el monitoreo climático.

## Metodología

### Exploración y Limpieza de Datos
- Dataset: `Precipitaciondt.csv`
- Imputación de valores nulos.
- Codificación de variables categóricas con **One-Hot Encoding**.

### Preparación de Datos
- Cálculo de promedios mensuales por cuenca.
- Variables predictoras (X): precipitaciones de enero a noviembre.
- Variable objetivo (Y): precipitación de diciembre.

### Entrenamiento
- División de datos: 80% entrenamiento, 20% prueba.
- Modelos probados:
    - Regresión Lineal (modelo base)
    - Random Forest Regressor ajustado con **GridSearchCV** (modelo final)

## Evaluación
El modelo final (Random Forest Regressor) obtuvo los siguientes resultados:
- **RMSE**: 7.69
- **R²**: 0.77

Estos resultados muestran una buena capacidad de predicción, indicando que el modelo es adecuado para la tarea planteada.

---


