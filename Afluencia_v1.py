import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

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
