import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand Forecasting App",
    layout="wide"
)

# --- Helper Functions (with Caching for performance) ---

@st.cache_data
def load_data(ingresos_upload, egresos_upload):
    """Loads data from user-uploaded files."""
    try:
        df_ingresos = pd.read_csv(ingresos_upload, parse_dates=['FechaEmision', 'RecibidaEl', 'TaxDate'])
        df_egresos = pd.read_csv(egresos_upload, parse_dates=['GENERADA_EL_FECHA_HORA', 'FECHA_ENTREGA'])
        return df_ingresos, df_egresos
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def create_features(_df_demand):
    """
    Takes the raw demand data and transforms it into a weekly time series DataFrame
    with engineered features (lags, rolling means, time components, hierarchy).
    """
    df_ts = _df_demand[['GENERADA_EL_FECHA_HORA', 'BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'CANTIDAD']].copy()
    df_ts.rename(columns={'GENERADA_EL_FECHA_HORA': 'fecha'}, inplace=True)

    df_weekly = df_ts.set_index('fecha').groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).resample('W')['CANTIDAD'].sum().reset_index()
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

    return df_featured.dropna()

@st.cache_data
def train_and_evaluate_model(_df_model_data):
    """Splits data, trains a LightGBM model, and evaluates its performance."""
    validation_weeks = 12
    split_date = _df_model_data['fecha'].max() - pd.Timedelta(weeks=validation_weeks)
    train = _df_model_data[_df_model_data['fecha'] <= split_date]
    val = _df_model_data[_df_model_data['fecha'] > split_date]

    FEATURES = [
        'year', 'month', 'week_of_year', 'quarter', 'lag_1', 'lag_2', 'lag_4', 'lag_52',
        'rolling_mean_4', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA'
    ]
    TARGET = 'cantidad_semanal'
    CATEGORICAL_FEATURES = ['DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])

    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    df_results = val[['fecha', 'SKU_ALTERNO', 'BODEGA_ORIGEN_DESC']].copy()
    df_results['actual_sales'] = y_val
    df_results['predicted_sales'] = predictions

    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    return mae, rmse, df_results, feature_importance_df, model, FEATURES, CATEGORICAL_FEATURES

def create_time_features_for_future(df):
    """Creates time series features from a datetime index for future dates."""
    df_out = df.copy()
    df_out['year'] = df_out['fecha'].dt.year
    df_out['month'] = df_out['fecha'].dt.month
    df_out['week_of_year'] = df_out['fecha'].dt.isocalendar().week.astype(int)
    df_out['quarter'] = df_out['fecha'].dt.quarter
    return df_out

# *** FIX IS HERE: This is the new, high-performance function ***
@st.cache_data
def generate_future_forecast(_final_model, _df_model_data, forecast_weeks, _features, _categorical_features):
    """Generates an efficient, iterative forecast using lookups instead of merges."""
    
    # Use a multi-index for fast lookups
    history = _df_model_data.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'])
    
    # Get unique identifiers for which to generate forecasts
    identifiers = history.index.unique()
    
    future_df_list = []
    
    for date in pd.date_range(start=history['fecha'].max() + pd.Timedelta(weeks=1), periods=forecast_weeks, freq='W'):
        # Prepare the DataFrame for the current future week
        future_step_df = pd.DataFrame(index=identifiers).reset_index()
        future_step_df['fecha'] = date
        
        # Create time-based features
        future_step_df = create_time_features_for_future(future_step_df)
        
        # Add static categorical features
        static_features = history[_categorical_features].groupby(level=[0, 1]).first()
        future_step_df = future_step_df.merge(static_features, left_on=['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'], right_index=True)

        # --- Efficient Feature Calculation using GroupBy and direct lookups ---
        grouped_history = history.groupby(level=[0, 1])['cantidad_semanal']

        future_step_df['lag_1'] = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).index.map(grouped_history.last())
        future_step_df['lag_2'] = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).index.map(grouped_history.nth(-2))
        future_step_df['lag_4'] = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).index.map(grouped_history.nth(-4))
        future_step_df['lag_52'] = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).index.map(grouped_history.nth(-52))
        future_step_df['rolling_mean_4'] = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']).index.map(grouped_history.rolling(4).mean().groupby(level=[0, 1]).last())

        future_step_df.fillna(0, inplace=True)
        
        # Predict and store
        predictions = _final_model.predict(future_step_df[_features])
        future_step_df['cantidad_semanal'] = np.maximum(0, predictions)
        
        # Append the new prediction to history for the next iteration
        new_history_row = future_step_df.set_index(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'])
        history = pd.concat([history, new_history_row])
        
        future_df_list.append(future_step_df)
        
    return pd.concat(future_df_list)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("Please upload the `INGRESOS` and `EGRESOS` CSV files.")
    uploaded_ingresos_file = st.file_uploader("Upload INGRESOS", type=['csv'])
    uploaded_egresos_file = st.file_uploader("Upload EGRESOS", type=['csv'])

# --- Main App Interface ---
# (The rest of the app's code is unchanged, as the fix was in the helper function)
st.title("📦 Demand Forecasting for Beds & Mattresses")

if uploaded_ingresos_file is not None and uploaded_egresos_file is not None:
    df_ingresos, df_egresos = load_data(uploaded_ingresos_file, uploaded_egresos_file)
    df_demand = df_egresos[(df_egresos['TIPO_DESC'] == 'Egreso') & (df_egresos['ESTADO_DESCRIP'] == 'Cerrada')].copy()

    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Model Training & Evaluation", "📈 Generate Forecast"])

    with tab1:
        st.header("Exploratory Data Analysis")
        st.subheader("Data Previews")
        col1, col2 = st.columns(2)
        with col1: st.write(f"**INGRESOS:** `{df_ingresos.shape[0]}` rows"); st.dataframe(df_ingresos.head())
        with col2: st.write(f"**EGRESOS:** `{df_egresos.shape[0]}` rows"); st.dataframe(df_egresos.head())

        st.sidebar.divider()
        st.sidebar.header("2. Dashboard Filters")
        all_bodegas = sorted(df_demand['BODEGA_ORIGEN_DESC'].unique())
        selected_bodegas = st.sidebar.multiselect('Select Warehouse(s)', options=all_bodegas, default=all_bodegas)
        if not selected_bodegas: st.warning("Please select at least one warehouse."); st.stop()
        filtered_demand = df_demand[df_demand['BODEGA_ORIGEN_DESC'].isin(selected_bodegas)]
        
        st.subheader("High-Level Metrics")
        total_quantity_sold=filtered_demand['CANTIDAD'].sum(); num_unique_skus=filtered_demand['SKU_ALTERNO'].nunique(); start_date=filtered_demand['GENERADA_EL_FECHA_HORA'].min().date(); end_date=filtered_demand['GENERADA_EL_FECHA_HORA'].max().date()
        kpi1, kpi2, kpi3 = st.columns(3); kpi1.metric("Total Quantity Sold", f"{int(total_quantity_sold):,}"); kpi2.metric("Unique SKUs Sold", f"{num_unique_skus:,}"); kpi3.metric("Date Range", f"{start_date} to {end_date}")

        st.subheader("Demand Analysis Visualizations"); sns.set_style("whitegrid")
        weekly_demand = filtered_demand.set_index('GENERADA_EL_FECHA_HORA').resample('W')['CANTIDAD'].sum()
        fig1, ax1 = plt.subplots(figsize=(12, 5)); sns.lineplot(data=weekly_demand, ax=ax1, lw=2); ax1.set_title("Total Weekly Demand", fontsize=16); st.pyplot(fig1)

    with tab2:
        st.header("Model Training and Evaluation")
        if st.button("🚀 Run Model Training & Evaluation", key='train_model'):
            with st.spinner("Processing data and training model..."):
                df_model_data = create_features(df_demand)
                mae, rmse, df_results, feat_imp, model, features, cat_features = train_and_evaluate_model(df_model_data)
                st.session_state.update({
                    'model_trained': True, 'df_model_data': df_model_data, 'mae': mae, 'rmse': rmse, 
                    'df_results': df_results, 'feat_imp': feat_imp, 'features': features, 'cat_features': cat_features
                })
        
        if 'model_trained' in st.session_state:
            st.success("Model training and evaluation complete!")
            st.subheader("Model Performance Metrics")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("MAE", f"{st.session_state['mae']:.2f}"); metric_col2.metric("RMSE", f"{st.session_state['rmse']:.2f}")
            st.info(f"💡 On average, the model's weekly forecast for a single SKU is off by approximately **{st.session_state['mae']:.2f} units**.")
            
            st.subheader("Validation: Actual vs. Predicted Sales")
            actuals_agg = st.session_state['df_results'].groupby('fecha')['actual_sales'].sum()
            predictions_agg = st.session_state['df_results'].groupby('fecha')['predicted_sales'].sum()
            fig_val, ax_val = plt.subplots(figsize=(12, 6)); ax_val.plot(actuals_agg.index, actuals_agg.values, label='Actual', marker='o'); ax_val.plot(predictions_agg.index, predictions_agg.values, label='Predicted', marker='x', linestyle='--'); ax_val.set_title('Total Weekly Sales (Validation Set)'); ax_val.legend(); st.pyplot(fig_val)
            
            st.subheader("Model Feature Importance")
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=st.session_state['feat_imp'].head(15), palette='viridis', ax=ax_imp, hue='feature', legend=False)
            ax_imp.set_title('Top 15 Most Important Features'); st.pyplot(fig_imp)

    with tab3:
        st.header("Generate Future Forecast")
        if 'model_trained' in st.session_state:
            forecast_weeks = st.slider("Forecast Horizon (Weeks)", 4, 52, 12, 1)
            if st.button("📈 Generate Forecast", key='generate_forecast'):
                with st.spinner("Retraining final model and generating forecast..."):
                    df_full_data = st.session_state['df_model_data']
                    X_full, y_full = df_full_data[st.session_state['features']], df_full_data['cantidad_semanal']
                    final_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
                    final_model.fit(X_full, y_full, categorical_feature=st.session_state['cat_features'])
                    forecast_df = generate_future_forecast(final_model, df_full_data, forecast_weeks, st.session_state['features'], st.session_state['cat_features'])
                    st.session_state['forecast_df'] = forecast_df

            if 'forecast_df' in st.session_state:
                st.success("Forecast generated successfully!")
                st.subheader("Historical Sales vs. Future Forecast")
                hist_agg = st.session_state['df_model_data'].groupby('fecha')['cantidad_semanal'].sum()
                forecast_agg = st.session_state['forecast_df'].groupby('fecha')['cantidad_semanal'].sum()
                
                fig_fc, ax_fc = plt.subplots(figsize=(12, 6)); ax_fc.plot(hist_agg.index, hist_agg.values, label='Historical'); ax_fc.plot(forecast_agg.index, forecast_agg.values, label='Forecasted', linestyle='--', marker='o'); ax_fc.axvline(hist_agg.index.max(), color='red', linestyle=':', label='Forecast Start'); ax_fc.set_title('Total Weekly Sales: Historical vs. Forecast'); ax_fc.legend(); st.pyplot(fig_fc)

                st.subheader("Forecast Data")
                forecast_display_cols = ['fecha', 'BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'cantidad_semanal']
                st.dataframe(st.session_state['forecast_df'][forecast_display_cols])
                st.download_button("📥 Download Forecast as CSV", convert_df_to_csv(st.session_state['forecast_df']), f"weekly_forecast_{forecast_weeks}_weeks.csv", 'text/csv')
        else:
            st.info("Please train a model in the 'Model Training & Evaluation' tab first.")

else:
    st.markdown("This application provides a comprehensive analysis and forecast of product demand. **To begin, please upload your data files using the sidebar on the left.**")
    st.info("Awaiting CSV file uploads...")

st.sidebar.divider()
st.sidebar.markdown("**Developed by:**  \nAntonio Medrano  \n*Data Scientist, CepSA*")
