#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
import os
import json
import gzip
import pickle
from glob import glob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error,
)

# Leer los datos
def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, compression='zip', index_col = False)
    df_test = pd.read_csv(test_path, compression='zip', index_col = False)
    return df_train, df_test

#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
def preprocess_data(df):
    df['Age'] = 2021 - df['Year']
    df.drop(columns=['Year', 'Car_Name'], inplace = True)
    return df
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
def divide_data(df_train, df_test):
    x_train = df_train.drop(columns=['Present_Price'])
    y_train = df_train['Present_Price']
    x_test = df_test.drop(columns=['Present_Price'])
    y_test = df_test['Present_Price']
    return x_train, y_train, x_test, y_test
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
def create_pipeline(x_train):
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = [x for x in x_train.columns if x not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression()),
    ])
    return pipeline
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
def optimize_hyperparameters(pipeline, x_train, y_train):
    total_features = pipeline.named_steps['preprocessor'].fit_transform(x_train).shape[1]

    param_grid = {
        'feature_selection__k': list(range(1, total_features + 1)),
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    return grid_search
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(model, f)
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
def calculate_and_save_metrics(dataset_type, y_true, y_pred, output_path):
    r2 = round(r2_score(y_true, y_pred), 4)
    mse = round(mean_squared_error(y_true, y_pred), 4)
    mad = round(median_absolute_error(y_true, y_pred), 4)
    metrics = {
        'type': 'metrics',
        'dataset': dataset_type,
        'r2': r2,
        'mse': mse,
        'mad': mad
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

if __name__ == "__main__":
    # Rutas de los archivos
    train_path = "files/input/train_data.csv.zip"
    test_path = "files/input/test_data.csv.zip"
    model_path = "files/models/model.pkl.gz"
    metrics_path = "files/output/metrics.json"

    # Cargar datos
    df_train, df_test = load_data(train_path, test_path)

    # Preprocesar datos
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    # Dividir datos
    x_train, y_train, x_test, y_test = divide_data(df_train, df_test)

    # Crear pipeline
    pipeline = create_pipeline(x_train)

    # Optimizar hiperparametros
    best_model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Guardar modelo
    save_model(best_model, model_path)

    # Calcular y guardar metricas para el conjunto de entrenamiento
    y_train_pred = best_model.predict(x_train)
    calculate_and_save_metrics('train', y_train, y_train_pred, metrics_path)

    # Calcular y guardar metricas para el conjunto de prueba
    y_test_pred = best_model.predict(x_test)
    calculate_and_save_metrics('test', y_test, y_test_pred, metrics_path)