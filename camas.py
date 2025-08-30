import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Pronóstico de Demanda",
    page_icon="🛏️",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded file object."""
    return pd.read_csv(uploaded_file, parse_dates=True)

@st.cache_data
def preprocess_and_feature_engineer(_df_demand): # The underscore prevents streamlit from hashing the input df name
    """Aggregates data weekly and creates all necessary features for the model."""
    df_demand_filtered = _df_demand[
        (_df_demand['TIPO_DESC'] == 'Egreso') &
        (_df_demand['ESTADO_DESCRIP'] == 'Cerrada')
    ].copy()

    # 1. Aggregate to weekly level
    df_ts = df_demand_filtered[['GENERADA_EL_FECHA_HORA', 'BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'CANTIDAD']].copy()
    df_ts['fecha'] = pd.to_datetime(df_ts['GENERADA_EL_FECHA_HORA']).dt.date
    df_ts['fecha'] = pd.to_datetime(df_ts['fecha'])
    
    df_weekly = df_ts.set_index('fecha').groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).resample('W')['CANTIDAD'].sum().reset_index()
    df_weekly.rename(columns={'CANTIDAD': 'cantidad_semanal'}, inplace=True)

    # 2. Create Time Features
    df_weekly['year'] = df_weekly['fecha'].dt.year
    df_weekly['month'] = df_weekly['fecha'].dt.month
    df_weekly['week_of_year'] = df_weekly['fecha'].dt.isocalendar().week.astype(int)
    df_weekly['quarter'] = df_weekly['fecha'].dt.quarter

    # 3. Create Lag and Rolling Features
    df_weekly.sort_values(by=['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'fecha'], inplace=True)
    grouped = df_weekly.groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'])['cantidad_semanal']
    df_weekly['lag_1'] = grouped.shift(1)
    df_weekly['lag_2'] = grouped.shift(2)
    df_weekly['lag_4'] = grouped.shift(4)
    df_weekly['lag_52'] = grouped.shift(52)
    df_weekly['rolling_mean_4'] = grouped.transform(lambda x: x.rolling(window=4).mean())
    
    # 4. Add Hierarchy Features
    hierarchy_cols = ['SKU_ALTERNO', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    sku_attributes = _df_demand[hierarchy_cols].drop_duplicates(subset=['SKU_ALTERNO'])
    df_final_features = pd.merge(df_weekly, sku_attributes, on='SKU_ALTERNO', how='left')
    
    # Convert categories and handle potential NAs from merge
    for col in ['DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']:
        df_final_features[col] = df_final_features[col].astype('category')
        
    return df_final_features

@st.cache_resource
def train_final_model(_df_features): # Underscore to avoid hashing issues
    """Trains and returns the final LightGBM model on all available data."""
    df_model_data = _df_features.dropna()
    
    FEATURES = [
        'year', 'month', 'week_of_year', 'quarter', 'lag_1', 'lag_2', 'lag_4', 'lag_52',
        'rolling_mean_4', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA'
    ]
    TARGET = 'cantidad_semanal'
    CATEGORICAL_FEATURES = ['DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']

    X_full = df_model_data[FEATURES]
    y_full = df_model_data[TARGET]

    final_model = lgb.LGBMRegressor(n_estimators=61, learning_rate=0.05, random_state=42, n_jobs=-1)
    final_model.fit(X_full, y_full, categorical_feature=CATEGORICAL_FEATURES)
    return final_model

def generate_forecast(_model, _historical_data, forecast_weeks):
    """Generates a future forecast using an iterative approach."""
    FEATURES = _model.feature_name_
    CATEGORICAL_FEATURES = [col for col in FEATURES if _historical_data[col].dtype.name == 'category']

    last_date = _historical_data['fecha'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_weeks, freq='W')
    
    current_history = _historical_data.sort_values('fecha').groupby('SKU_ALTERNO').tail(52)
    future_df = pd.DataFrame()

    for date in future_dates:
        future_step = pd.DataFrame({'fecha': [date]})
        future_step['year'] = future_step['fecha'].dt.year
        future_step['month'] = future_step['fecha'].dt.month
        future_step['week_of_year'] = future_step['fecha'].dt.isocalendar().week.astype(int)
        future_step['quarter'] = future_step['fecha'].dt.quarter
        future_step['key'] = 1

        unique_skus = current_history[['SKU_ALTERNO'] + CATEGORICAL_FEATURES].drop_duplicates()
        unique_skus['key'] = 1
        
        future_step_df = pd.merge(unique_skus, future_step, on='key').drop('key', axis=1)
        
        sku_to_last_value = current_history.groupby('SKU_ALTERNO')['cantidad_semanal']
        future_step_df['lag_1'] = future_step_df['SKU_ALTERNO'].map(sku_to_last_value.nth(-1))
        future_step_df['lag_2'] = future_step_df['SKU_ALTERNO'].map(sku_to_last_value.nth(-2))
        future_step_df['lag_4'] = future_step_df['SKU_ALTERNO'].map(sku_to_last_value.nth(-4))
        future_step_df['lag_52'] = future_step_df['SKU_ALTERNO'].map(sku_to_last_value.nth(-52))
        last_rolling_mean = sku_to_last_value.rolling(4).mean().groupby('SKU_ALTERNO').last()
        future_step_df['rolling_mean_4'] = future_step_df['SKU_ALTERNO'].map(last_rolling_mean)

        future_step_df = future_step_df.fillna(0)

        predictions = _model.predict(future_step_df[FEATURES])
        future_step_df['cantidad_semanal'] = np.maximum(0, predictions)

        current_history = pd.concat([current_history, future_step_df])
        future_df = pd.concat([future_df, future_step_df])
        
    return future_df

# --- Main App ---
st.title("🛏️ Prototipo de Pronóstico de Demanda")
st.write("Esta aplicación utiliza un modelo de Machine Learning (LightGBM) para pronosticar la demanda semanal de camas y colchones.")

# --- Sidebar for File Uploads ---
st.sidebar.header("1. Cargar Archivos de Datos")
uploaded_egresos_file = st.sidebar.file_uploader(
    "Cargar archivo de EGRESOS (Requerido)", 
    type="csv", 
    help="Este es el archivo de salidas de inventario, que representa la demanda histórica."
)
uploaded_ingresos_file = st.sidebar.file_uploader(
    "Cargar archivo de INGRESOS (Opcional)", 
    type="csv",
    help="Este archivo de entradas de inventario se puede usar para análisis adicionales."
)

if uploaded_egresos_file is not None:
    # --- Data Processing and Modeling ---
    df_demand_raw = load_data(uploaded_egresos_file)
    df_history = preprocess_and_feature_engineer(df_demand_raw)
    final_model = train_final_model(df_history)

    # --- Sidebar for Forecast Controls ---
    st.sidebar.header("2. Configurar Pronóstico")
    weeks_to_forecast = st.sidebar.slider("Semanas a Pronosticar", min_value=4, max_value=52, value=12, step=4)
    
    st.sidebar.header("3. Filtros de Visualización")
    marcas = sorted(df_history['DESC_MARCA'].astype(str).unique().tolist())
    familias = sorted(df_history['FAMILIA'].astype(str).unique().tolist())
    selected_marcas = st.sidebar.multiselect("Filtrar por Marca", marcas, default=marcas)
    selected_familias = st.sidebar.multiselect("Filtrar por Familia", familias, default=familias)

    # --- Generate and Display Forecast ---
    with st.spinner('Generando pronóstico...'):
        forecast_df = generate_forecast(final_model, df_history, weeks_to_forecast)

    if not selected_marcas: selected_marcas = marcas
    if not selected_familias: selected_familias = familias

    history_filtered = df_history[df_history['DESC_MARCA'].isin(selected_marcas) & df_history['FAMILIA'].isin(selected_familias)]
    forecast_filtered = forecast_df[forecast_df['DESC_MARCA'].isin(selected_marcas) & forecast_df['FAMILIA'].isin(selected_familias)]

    total_forecasted_units = forecast_filtered['cantidad_semanal'].sum()
    st.header(f"Pronóstico para las próximas {weeks_to_forecast} semanas")
    st.metric("Total de Unidades Pronosticadas (Filtro Actual)", f"{total_forecasted_units:,.0f}")

    hist_agg = history_filtered.groupby('fecha')['cantidad_semanal'].sum().reset_index()
    hist_agg['Tipo'] = 'Ventas Históricas'
    forecast_agg = forecast_filtered.groupby('fecha')['cantidad_semanal'].sum().reset_index()
    forecast_agg['Tipo'] = 'Pronóstico'
    plot_df = pd.concat([hist_agg, forecast_agg])

    fig = px.line(plot_df, x='fecha', y='cantidad_semanal', color='Tipo', title="Ventas Semanales: Histórico vs. Pronóstico", labels={'fecha': 'Semana', 'cantidad_semanal': 'Cantidad de Unidades'}, markers=True)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver datos detallados del pronóstico"):
        display_df = forecast_filtered[['fecha', 'SKU_ALTERNO', 'DESC_MARCA', 'FAMILIA', 'cantidad_semanal']].rename(columns={'fecha': 'Semana', 'cantidad_semanal': 'Cantidad Pronosticada'}).sort_values(by=['Semana', 'SKU_ALTERNO'])
        st.dataframe(display_df)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Pronóstico como CSV", csv, f'pronostico_{weeks_to_forecast}_semanas.csv', 'text/csv')

else:
    st.info("👋 ¡Bienvenido! Por favor, cargue el archivo de **EGRESOS** para comenzar a generar el pronóstico.")

# --- Optional Analysis of INGRESOS file ---
if uploaded_ingresos_file is not None:
    with st.expander("Análisis Exploratorio del Archivo de Ingresos"):
        st.info("Nota: El archivo de ingresos no se utiliza en el modelo de pronóstico de demanda, pero se muestra aquí para análisis de contexto, como el volumen de compras por proveedor.")
        df_ingresos = load_data(uploaded_ingresos_file)
        st.subheader("Vista Previa de los Datos de Ingresos")
        st.dataframe(df_ingresos.head())

        st.subheader("Top 10 Proveedores por Volumen de Recepciones")
        proveedor_counts = df_ingresos['Proveedor'].value_counts().nlargest(10)
        st.bar_chart(proveedor_counts)
