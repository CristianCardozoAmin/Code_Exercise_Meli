# Opportunities@MeLi - Ejercicio Técnico de Machine Learning

## Introducción y objetivo
El objetivo principal de este ejercicio es desarrollar un algoritmo que prediga si un ítem publicado en MercadoLibre es **nuevo (0)** o **usado (1)** a partir de la información disponible en los datos.

---

## Estructura del Proyecto

### 📁 Notebooks
- `exploratory_analysis.ipynb`: Análisis exploratorio inicial (EDA).
- `model_01.ipynb`: Desarrollo del modelo usando **XGBoost**.
- `model_02.ipynb`: Desarrollo del modelo usando **LightGBM**.

### 📁 Scripts Python
- `Preprocessing.py`: Transformación inicial del JSON en datos tabulares manejables.
- `Transformation.py`: Aplicación de transformaciones específicas identificadas en el EDA.
- `model_processing.py`: Preparación final de los datos (normalización, codificación dummy y conversión de booleanos).
- `new_or_used.py`: Código para desplegar el modelo en producción usando FastAPI.

---

## Descripción de los datos
Los datos iniciales se obtienen de una API en formato JSON, que contiene estructuras anidadas complejas. Se utilizó una función recursiva implementada en `Preprocessing.py` para convertir estas estructuras en columnas útiles para Machine Learning.

Luego, mediante un Análisis Exploratorio Automático (EDA), se identificaron variables útiles, completitud de datos y transformaciones necesarias.

---

## Transformaciones realizadas
Las transformaciones claves implementadas en `Transformation.py` fueron:

- **Conversión de moneda:** Homogeneización a pesos argentinos considerando tasas históricas.
- **Extracción de componentes temporales:** Separación de fecha y hora en variables numéricas individuales (año, mes, día, semana ISO, día de la semana, hora, minuto, segundo).
- **Transformación de variables categóricas:**
  - Conversión a booleanas si la cantidad de nulos es alta.
  - Reducción de categorías dominantes a variables binarias.
  - Transformación de variables ordinales a numéricas.
- **Extracción de coordenadas geográficas:** Uso de la API de MercadoLibre para obtener latitud y longitud.
- **Eliminación de variables:**
  - Variables con más del 95% de valores nulos.
  - Variables con alta cardinalidad o identificadores únicos (URLs).
- **Conversión de dimensiones de fotos:** Separación de dimensiones en alto y ancho.

Estas transformaciones se complementan con las preparaciones finales definidas en `model_processing.py`, incluyendo estandarización y codificación dummy.

---

## Modelos utilizados
Para resolver el problema de clasificación se utilizaron dos modelos robustos ante valores faltantes y multicolinealidad:

- **Extreme Gradient Boosting (XGBoost)**
- **Light Gradient Boosting (LightGBM)**

Ambos modelos se optimizaron utilizando `GridSearchCV` y la métrica principal solicitada fue el **Accuracy**.

---

## Desempeño de los Modelos

| Modelo          | Accuracy | F1-score | AUC ROC |
|-----------------|----------|----------|---------|
| XGBoost         | **0.881**    | **0.875**    | **0.95** |
| LightGBM        | 0.8733   | 0.8675   | 0.95 |

Se recomienda como métrica secundaria el área bajo la curva ROC (**AUC**), por su capacidad intuitiva para evaluar el desempeño del modelo en términos de falsos positivos y verdaderos positivos.

---

## Despliegue
El modelo final fue desplegado usando **FastAPI** para ofrecer una API sencilla que consume una base de datos como input y retorna la predicción correspondiente. El código para esta tarea se encuentra en:

- `api_model_serving.py`

