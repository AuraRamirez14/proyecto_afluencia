import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

st.write(''' Predicción de afluencia en el Sistema de Transporte Colectivo Metro de la Ciudad de México ''')
st.image("metroo.png", caption="Afluencia en el STC Metro.")

st.header('Datos a evaluar')

def user_input_features():
    # Entrada de variables
    anio = st.number_input('Año (yyyy):', min_value=2010, max_value=2025, value=2020, step=1)
    codificacion_mes = st.number_input('Mes:', min_value=1, max_value=12, value=1, step=1)
    diaSemana = st.number_input('Día de la semana:', min_value=1, max_value=7, value=1, step=1)
    codificacion_linea = st.number_input('Línea:', min_value=1, max_value=12, value=1, step=1)
    codificacion_estacion = st.number_input('Estación:', min_value=1, max_value=198, value=1, step=1)

    user_input_data = {
        'anio': anio,
        'codificacion_mes': codificacion_mes,
        'diaSemana': diaSemana,
        'codificacion_linea': codificacion_linea,
        'codificacion_estacion': codificacion_estacion
    }
    
    return pd.DataFrame(user_input_data, index=[0])

df = user_input_features()

# Leer datos comprimidos
afluencia = pd.read_csv("afluencia_limpio.csv.gz", encoding='latin-1')
X = afluencia.drop(columns='afluencia')
Y = afluencia['afluencia']

# Entrenar modelo de regresión (puedes guardarlo luego con joblib)
regressor = DecisionTreeRegressor(criterion='squared_error',   # error cuadrático para regresión
    max_depth=7,                 # limita profundidad
    min_samples_split=10,        # mínimo para dividir un nodo
    min_samples_leaf=4,          # mínimo en cada hoja
    max_leaf_nodes=20,           # limita número de hojas
    ccp_alpha=0.005,             # poda ligera
    random_state=0
regressor.fit(X, Y)

# Predicción numérica exacta
prediction = regressor.predict(df)

st.subheader('Predicción')
st.metric(label="Afluencia estimada", value=int(prediction[0]))
