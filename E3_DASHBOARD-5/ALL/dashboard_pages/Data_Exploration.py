import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy.stats import f_oneway
import numpy as np
import statsmodels.api as sm
import joblib

st.title("📊 Exploración de datos")
st.write("Esta página muestra la proporción de interacciones por candidato en función de la plataforma seleccionada.")

# Cargar base
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import timedelta

# Configuración de la página
st.title("Proporción de Interacciones por Candidato")
st.write("""
En esta sección se presenta un gráfico de proporción de interacciones por candidato, basado en los datos de la base `Pollster_Data`.
El gráfico permite filtrar por plataforma y utiliza únicamente los datos recopilados antes del 28 de mayo.
""")

# Cargar los datos (modifica la ruta según la ubicación de tu archivo)
@st.cache_data
def load_data():
    pollster_data = pd.read_csv("ALL/final_datasets/Pollster_vs_Metrics.csv")  # Ajusta el nombre y ubicación del archivo
    # Convertir la columna de fecha a tipo datetime
    pollster_data['date'] = pd.to_datetime(pollster_data['date'])
    # Filtrar los datos antes del 28 de mayo
    filtered_data = pollster_data[pollster_data['date'] < "2024-05-28"]
    return filtered_data

pollster_data = load_data()

# Filtro para elegir la plataforma
platform = st.selectbox(
    "Selecciona la Plataforma:",
    options=["twitter", "instagram", "facebook", "youtube"]
)

# Mapeo de las columnas por plataforma
columns_mapping = {
    "twitter": [
        "claudia_x_num_interaction_share",
        "xochitl_x_num_interaction_share",
        "maynez_x_num_interaction_share",
    ],
    "instagram": [
        "claudia_x_num_interaction_share",
        "xochitl_x_num_interaction_share",
        "maynez_x_num_interaction_share",
    ],
    "facebook": [
        "claudia_x_num_interaction_share",
        "xochitl_x_num_interaction_share",
        "maynez_x_num_interaction_share",
    ],
    "youtube": [
        "claudia_x_num_interaction_share",
        "xochitl_x_num_interaction_share",
        "maynez_x_num_interaction_share",
    ]
}

# Seleccionar las columnas correspondientes a la plataforma elegida
selected_columns = columns_mapping[platform]

# Extraer datos relevantes
proportion_data = pollster_data[selected_columns]

# Renombrar columnas para mayor claridad en el gráfico
proportion_data.columns = ["Claudia", "Xóchitl", "Maynez"]

# Calcular la suma total de interacciones por candidato
proportion_sum = proportion_data.sum()

candidate_plot_colors = {
"Claudia": '#741D23',      
"Maynez": '#FF8300',      
"Xóchitl": '#1E75BC'       
}

# Crear el gráfico de pie
fig = px.pie(
    names=proportion_sum.index,
    values=proportion_sum.values,
    title=f"Proporción de Interacciones por Candidato en {platform.capitalize()}",
    color=proportion_sum.index,  # Opcional: Añade colores específicos si deseas
    hole=0.4,  # Gráfico de dona
    color_discrete_map=candidate_plot_colors
)

# Mostrar el gráfico en la app
st.plotly_chart(fig)










# Linea de Tendencia

# Cargar datos
pollster_data = pd.read_csv("ALL/final_datasets/Pollster_vs_Metrics_Predicted.csv")
pollster_data["date"] = pd.to_datetime(pollster_data["date"])

# Diccionario para renombrar métricas a nombres más amigables
metric_labels = {
    "claudia_voting_intention": "Claudia's Voting Intention",
    "xochitl_voting_intention": "Xóchitl's Voting Intention",
    "maynez_voting_intention": "Máynez's Voting Intention",
}

# Selección de métricas para el candidato actual
selected_metrics = ["claudia_voting_intention", "xochitl_voting_intention", "maynez_voting_intention"]

# Filtrar datos relevantes (asegurando crear una copia explícita)
filtered_data = pollster_data[["date"] + selected_metrics].copy()

# Filtrar datos antes de 2024-05-28
filtered_data = filtered_data[filtered_data["date"] < "2024-05-28"]

# Renombrar las métricas con nombres amigables
melted_data = filtered_data.melt(id_vars="date", var_name="Métrica", value_name="Valor").copy()
melted_data["Métrica"] = melted_data["Métrica"].map(metric_labels)

candidate_plot_colors = {
"Claudia's Voting Intention": '#741D23',      
"Máynez's Voting Intention": '#FF8300',      
"Xóchitl's Voting Intention": '#1E75BC'       
}

# Crear la gráfica con px.line
st.subheader("Tendencia de la Intención de Voto")
st.write("Con la intención de tener un parámetro de comparativa relacionado completamente con las elecciones, se importó el [Barómetro de Bloomberg](https://www.bloomberg.com/graphics/Mexico-Encuestas-Presidenciales-2024-ventaja-sheinbaum-galvez-veda/), que indica la **Intención de Voto** para cada candidato a partir de una ponderación de todas las encuestas realizadas a la población Mexicana.")
fig = px.line(
melted_data,
x="date",
y="Valor",
color="Métrica",
title=f"Tendencia de la Intención de Voto",
labels={"date": "Fecha", "Valor": "Valor", "Métrica": "Métrica"},
template="plotly_white",
color_discrete_map=candidate_plot_colors  # Apply the color dictionary
)

# Mostrar el gráfico
st.plotly_chart(fig)
