import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


st.write(''' Predicción de afluencia en el Sistema de Transporte Colectivo Metro de la Ciudad de México ''')
st.image("metroo.png", caption="Afluencia en el STC Metro.")
#st.imagen("tabladia", caption="Conversión a númerico del día de la semana")
#st.imagen("tablaestacion", caption="Conversión a númerico de la estación")
#st.imagen("tablalinea", caption="Conversión a númerico de la línea")

st.header('Datos a evaluar')

def user_input_features():
  # Entrada
  anio = st.number_input('Año(yyyy):', min_value=2010, max_value=2025, value = 2020, step = 1)
  codificacion_mes = st.number_input('Mes:', min_value=1, max_value=12, value = 1, step = 1)
  diaSemana = st.number_input('Día de la semana:', min_value=1, max_value=7, value = 1, step = 1)
  #esFestivo = st.number_input('Dia que marca si fue festivo:', min_value=0, max_value=1, value = 0, step = 1)
  #puntoInteres = st.number_input('Existe algún punto de interés:',min_value=0, max_value=10, value = 0, step = 1)
  codificacion_linea = st.number_input('Linea:', min_value=1, max_value=12, value = 1, step = 1)
  codificacion_estacion = st.number_input('Estación:', min_value=1, max_value=198, value = 1, step = 1)


  user_input_data = {'anio': anio,
                     'codificacion_mes': codificacion_mes,
                     'diaSemana': diaSemana,
                     #'esFestivo': esFestivo,
                     #'puntoInteres': puntoInteres,
                     'codificacion_linea': codificacion_linea,
                     'codificacion_estacion': codificacion_estacion}
                     
  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

#afluencia =  pd.read_csv('afluencia_limpio.csv', encoding='latin-1')
afluencia = pd.read_csv("afluencia_limpio.csv.gz", encoding='latin-1')
X = afluencia.drop(columns='afluencia')
Y = afluencia['afluencia']

classifier = DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=10, random_state=0)
classifier.fit(X, Y)

prediction = classifier.predict(df)
prediction_probabilities = classifier.predict_proba(df)

st.subheader('Predicción')
st.write('Dadas las variables que se seleccionaron, la afluencia predicha es:', prediction)
st.metric(label="Afluencia estimada", value=int(prediction[0]))
