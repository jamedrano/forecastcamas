import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📦",
    layout="wide"
)

# --- Helper Functions (with Caching for performance) ---

@st.cache_data
def load_data(file_upload):
    """Loads data from user-uploaded file, robustly detecting separator."""
    if file_upload is None:
        return None
    try:
        file_upload.seek(0)
        df = pd.read_csv(file_upload, sep=None, engine='python')
        # Parse dates after loading
        potential_date_cols = ['FechaEmision', 'RecibidaEl', 'TaxDate', 'GENERADA_EL_FECHA_HORA', 'FECHA_ENTREGA']
        for col in potential_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading or parsing the file: {e}")
        return None

@st.cache_data
def create_features(_df_demand):
    """
    Takes the raw demand data and transforms it into a weekly time series DataFrame
    with engineered features (lags, rolling means, time components, hierarchy).
    """
    df_ts = _df_demand[['GENERADA_EL_FECHA_HORA', 'BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'CANTIDAD']].copy()
    df_ts.rename(columns={'GENERADA_EL_FECHA_HORA': 'fecha'}, inplace=True)
    df_weekly = df_ts.set_index('fecha').groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).resample('W-SUN')['CANTIDAD'].sum().reset_index()
    df_weekly.rename(columns={'CANTIDAD': 'cantidad_semanal'}, inplace=True)
    df_weekly['year'] = df_weekly['fecha'].dt.year
    df_weekly['month'] = df_weekly['fecha'].dt.month
    df_weekly['week_of_year'] = df_weekly['fecha'].dt.isocalendar().week.astype(int)
    df_weekly['quarter'] = df_weekly['fecha'].dt.quarter
    df_weekly.sort_values(by=['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'fecha'], inplace=True)
    grouped = df_weekly.groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'])['cantidad_semanal']
    df_weekly['lag_1'] = grouped.shift(1)
    df_weekly['lag_2'] = grouped.shift(2)
    df_weekly['lag_4'] = grouped.shift(4)
    df_weekly['lag_52'] = grouped.shift(52)
    df_weekly['rolling_mean_4'] = grouped.transform(lambda x: x.shift(1).rolling(window=4).mean())
    hierarchy_cols = ['SKU_ALTERNO', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    sku_attributes = _df_demand[hierarchy_cols].drop_duplicates(subset=['SKU_ALTERNO'])
    df_featured = pd.merge(df_weekly, sku_attributes, on='SKU_ALTERNO', how='left')
    categorical_features = ['DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    for col in categorical_features:
        df_featured[col] = df_featured[col].astype('category')
    return df_featured

@st.cache_data
def train_and_evaluate_model(_df_featured):
    """Splits data, trains a LightGBM model, and evaluates its performance."""
    df_model_data = _df_featured.dropna()
    validation_weeks = 12
    split_date = df_model_data['fecha'].max() - pd.Timedelta(weeks=validation_weeks)
    train = df_model_data[df_model_data['fecha'] <= split_date]
    val = df_model_data[df_model_data['fecha'] > split_date]
    FEATURES = ['year', 'month', 'week_of_year', 'quarter', 'lag_1', 'lag_2', 'lag_4', 'lag_52', 'rolling_mean_4', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    TARGET = 'cantidad_semanal'
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    df_results = val[['fecha', 'SKU_ALTERNO', 'DESC_MARCA', 'FAMILIA']].copy()
    df_results['actual_sales'] = y_val
    df_results['predicted_sales'] = predictions
    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return mae, rmse, df_results, feature_importance_df, model, df_model_data

@st.cache_resource
def train_final_model(_df_model_data, _best_params):
    """Retrains the model on all data for production forecasting."""
    FEATURES = ['year', 'month', 'week_of_year', 'quarter', 'lag_1', 'lag_2', 'lag_4', 'lag_52', 'rolling_mean_4', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    TARGET = 'cantidad_semanal'
    X_full, y_full = _df_model_data[FEATURES], _df_model_data[TARGET]
    final_model = lgb.LGBMRegressor(**_best_params)
    final_model.fit(X_full, y_full, categorical_feature='auto')
    return final_model

@st.cache_data
def generate_future_forecast(_model, _historical_data, forecast_weeks):
    """Generates a future forecast using an iterative approach."""
    FEATURES = _model.feature_name_
    CATEGORICAL_FEATURES = [col for col in FEATURES if _historical_data[col].dtype.name == 'category']
    last_date = _historical_data['fecha'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_weeks, freq='W-SUN')
    current_history = _historical_data.sort_values('fecha').groupby('SKU_ALTERNO').tail(52)
    future_df = pd.DataFrame()
    for date in future_dates:
        future_step = pd.DataFrame({'fecha': [date]})
        future_step['year'], future_step['month'], future_step['week_of_year'], future_step['quarter'] = date.year, date.month, date.isocalendar().week, date.quarter
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
        numeric_features_to_fill = ['lag_1', 'lag_2', 'lag_4', 'lag_52', 'rolling_mean_4']
        future_step_df[numeric_features_to_fill] = future_step_df[numeric_features_to_fill].fillna(0)
        predictions = _model.predict(future_step_df[FEATURES])
        future_step_df['cantidad_semanal'] = np.maximum(0, predictions)
        current_history = pd.concat([current_history, future_step_df])
        future_df = pd.concat([future_df, future_step_df])
    return future_df

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("Please upload `INGRESOS` and `EGRESOS` CSV files.")
    uploaded_ingresos_file = st.file_uploader("Upload INGRESOS (Optional)", type=['csv'])
    uploaded_egresos_file = st.file_uploader("Upload EGRESOS (Required)", type=['csv'])

# --- Main App Interface ---
st.title("📦 Demand Forecasting for Beds & Mattresses")

if uploaded_egresos_file is not None:
    df_ingresos = load_data(uploaded_ingresos_file)
    df_egresos = load_data(uploaded_egresos_file)
    df_demand = df_egresos[(df_egresos['TIPO_DESC'] == 'Egreso') & (df_egresos['ESTADO_DESCRIP'] == 'Cerrada')].copy()

    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Model Training & Evaluation", "📈 Forecast"])

    with tab1:
        st.header("Exploratory Data Analysis")
        st.subheader("Data Previews")
        if df_ingresos is not None:
             st.write(f"**INGRESOS (Inflows):** `{df_ingresos.shape[0]}` rows, `{df_ingresos.shape[1]}` columns.")
             st.dataframe(df_ingresos.head())
        st.write(f"**EGRESOS (Outflows):** `{df_egresos.shape[0]}` rows, `{df_egresos.shape[1]}` columns.")
        st.dataframe(df_egresos.head())
        
        st.sidebar.divider()
        st.sidebar.header("2. Dashboard Filters")
        all_bodegas = sorted(df_demand['BODEGA_ORIGEN_DESC'].unique())
        selected_bodegas = st.sidebar.multiselect('Select Warehouse(s)', options=all_bodegas, default=all_bodegas)
        if not selected_bodegas: selected_bodegas = all_bodegas
        filtered_demand = df_demand[df_demand['BODEGA_ORIGEN_DESC'].isin(selected_bodegas)]

        st.subheader("High-Level Metrics")
        total_quantity_sold = filtered_demand['CANTIDAD'].sum()
        num_unique_skus = filtered_demand['SKU_ALTERNO'].nunique()
        start_date = filtered_demand['GENERADA_EL_FECHA_HORA'].min().date()
        end_date = filtered_demand['GENERADA_EL_FECHA_HORA'].max().date()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Quantity Sold", f"{int(total_quantity_sold):,}")
        kpi2.metric("Unique SKUs Sold", f"{num_unique_skus:,}")
        kpi3.metric("Date Range", f"{start_date} to {end_date}")

        st.subheader("Demand Analysis Visualizations")
        weekly_demand = filtered_demand.set_index('GENERADA_EL_FECHA_HORA').resample('W-SUN')['CANTIDAD'].sum().reset_index()
        fig1 = px.line(weekly_demand, x='GENERADA_EL_FECHA_HORA', y='CANTIDAD', title='Total Weekly Demand')
        st.plotly_chart(fig1, use_container_width=True)

        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            top_10_skus = filtered_demand.groupby('SKU_ALTERNO')['CANTIDAD'].sum().nlargest(10).reset_index()
            fig2 = px.bar(top_10_skus, y='SKU_ALTERNO', x='CANTIDAD', title='Top 10 SKUs by Quantity', orientation='h').update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig2, use_container_width=True)
        with col_viz2:
            top_10_brands = filtered_demand.groupby('DESC_MARCA')['CANTIDAD'].sum().nlargest(10).reset_index()
            fig3 = px.bar(top_10_brands, y='DESC_MARCA', x='CANTIDAD', title='Top 10 Brands by Quantity', orientation='h').update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.header("Model Training and Evaluation")
        if st.button("🚀 Run Model Training & Evaluation", key='train_model'):
            with st.spinner("Processing data and training model... This may take a few minutes."):
                df_featured = create_features(df_demand)
                mae, rmse, df_results, feature_importance_df, model, df_model_data = train_and_evaluate_model(df_featured)
                st.session_state.mae = mae
                st.session_state.rmse = rmse
                st.session_state.df_results = df_results
                st.session_state.feature_importance_df = feature_importance_df
                st.session_state.trained_model = model
                st.session_state.df_model_data = df_model_data
            st.success("Model training and evaluation complete!")
        
        if 'mae' in st.session_state:
            st.subheader("Model Performance Metrics")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Mean Absolute Error (MAE)", f"{st.session_state.mae:.2f}")
            metric_col2.metric("Root Mean Squared Error (RMSE)", f"{st.session_state.rmse:.2f}")
            st.info(f"💡 On average, the model's weekly forecast for a single SKU is off by approximately **{st.session_state.mae:.2f} units**.")
            
            st.subheader("Validation Period: Actual vs. Predicted Sales")
            plot_df = st.session_state.df_results.groupby('fecha').sum().reset_index()
            fig_val = px.line(plot_df, x='fecha', y=['actual_sales', 'predicted_sales'], title='Total Weekly Sales: Actual vs. Predicted (Validation Set)', markers=True)
            st.plotly_chart(fig_val, use_container_width=True)

            st.subheader("Model Feature Importance")
            fig_imp = px.bar(st.session_state.feature_importance_df.head(15), y='feature', x='importance', title='Top 15 Most Important Features', orientation='h').update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.header("Generate Future Forecast")
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model in the 'Model Training & Evaluation' tab first.")
        else:
            st.sidebar.divider()
            st.sidebar.header("3. Forecast Controls")
            forecast_weeks = st.sidebar.slider("Weeks to Forecast", min_value=4, max_value=52, value=12, step=4)

            if st.sidebar.button("Generate Forecast", key='generate_forecast'):
                with st.spinner("Retraining final model and generating forecast..."):
                    model_params = st.session_state.trained_model.get_params()
                    model_params['n_estimators'] = st.session_state.trained_model.best_iteration_
                    final_model = train_final_model(st.session_state.df_model_data, model_params)
                    st.session_state.forecast_df = generate_future_forecast(final_model, st.session_state.df_model_data, forecast_weeks)
                st.success("Forecast generated!")

            if 'forecast_df' in st.session_state:
                st.subheader(f"Forecasted Demand for the Next {len(st.session_state.forecast_df['fecha'].unique())} Weeks")
                
                # Use filters from sidebar
                forecast_filtered = st.session_state.forecast_df[st.session_state.forecast_df['DESC_MARCA'].isin(selected_marcas) & st.session_state.forecast_df['FAMILIA'].isin(selected_familias)]
                history_filtered = st.session_state.df_model_data[st.session_state.df_model_data['DESC_MARCA'].isin(selected_marcas) & st.session_state.df_model_data['FAMILIA'].isin(selected_familias)]
                
                hist_agg = history_filtered.groupby('fecha')['cantidad_semanal'].sum().reset_index()
                hist_agg['Type'] = 'Historical Sales'
                forecast_agg = forecast_filtered.groupby('fecha')['cantidad_semanal'].sum().reset_index()
                forecast_agg['Type'] = 'Forecasted Sales'
                
                plot_df = pd.concat([hist_agg, forecast_agg])
                fig_forecast = px.line(plot_df, x='fecha', y='cantidad_semanal', color='Type', title='Historical Sales vs. Future Forecast', markers=True)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                with st.expander("View Detailed Forecast Data"):
                    st.dataframe(forecast_filtered)
                    csv = forecast_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Filtered Forecast as CSV", csv, "forecast.csv", "text/csv")
else:
    st.info("Awaiting CSV file uploads...")
