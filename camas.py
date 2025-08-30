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
def create_features(df_demand):
    """
    Takes the raw demand data and transforms it into a weekly time series DataFrame
    with engineered features (lags, rolling means, time components, hierarchy).
    """
    # 1. Select and rename essential columns
    df_ts = df_demand[['GENERADA_EL_FECHA_HORA', 'BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'CANTIDAD']].copy()
    df_ts.rename(columns={'GENERADA_EL_FECHA_HORA': 'fecha'}, inplace=True)

    # 2. Aggregate to a weekly level
    df_weekly = df_ts.set_index('fecha') \
                     .groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO']) \
                     .resample('W')['CANTIDAD'].sum().reset_index()
    df_weekly.rename(columns={'CANTIDAD': 'cantidad_semanal'}, inplace=True)

    # 3. Create time-based features
    df_weekly['year'] = df_weekly['fecha'].dt.year
    df_weekly['month'] = df_weekly['fecha'].dt.month
    df_weekly['week_of_year'] = df_weekly['fecha'].dt.isocalendar().week.astype(int)
    df_weekly['quarter'] = df_weekly['fecha'].dt.quarter

    # 4. Create Lag and Rolling Features
    df_weekly.sort_values(by=['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO', 'fecha'], inplace=True)
    grouped = df_weekly.groupby(['BODEGA_ORIGEN_DESC', 'SKU_ALTERNO'])['cantidad_semanal']
    df_weekly['lag_1'] = grouped.shift(1)
    df_weekly['lag_2'] = grouped.shift(2)
    df_weekly['lag_4'] = grouped.shift(4)
    df_weekly['lag_52'] = grouped.shift(52)
    df_weekly['rolling_mean_4'] = grouped.transform(lambda x: x.shift(1).rolling(window=4).mean())

    # 5. Add Product Hierarchy Features
    hierarchy_cols = ['SKU_ALTERNO', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    sku_attributes = df_demand[hierarchy_cols].drop_duplicates(subset=['SKU_ALTERNO'])
    df_featured = pd.merge(df_weekly, sku_attributes, on='SKU_ALTERNO', how='left')

    # Convert categoricals for LightGBM
    categorical_features = ['DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA']
    for col in categorical_features:
        df_featured[col] = df_featured[col].astype('category')

    return df_featured.dropna()

@st.cache_data
def train_and_evaluate_model(_df_model_data):
    """
    Splits data, trains a LightGBM model, and evaluates its performance.
    The underscore in _df_model_data is a convention to signal that the input is being cached.
    """
    # 1. Define Split Date
    validation_weeks = 12
    split_date = _df_model_data['fecha'].max() - pd.Timedelta(weeks=validation_weeks)
    train = _df_model_data[_df_model_data['fecha'] <= split_date]
    val = _df_model_data[_df_model_data['fecha'] > split_date]

    # 2. Define Features and Target
    FEATURES = [
        'year', 'month', 'week_of_year', 'quarter', 'lag_1', 'lag_2', 'lag_4', 'lag_52',
        'rolling_mean_4', 'DESC_MARCA', 'TIPO_PRODUCTO', 'FAMILIA', 'SUB_FAMILIA'
    ]
    TARGET = 'cantidad_semanal'

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    # 3. Instantiate and Train the Model
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # 4. Make Predictions and Evaluate
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    # 5. Prepare Results for Visualization
    df_results = val[['fecha', 'SKU_ALTERNO', 'BODEGA_ORIGEN_DESC']].copy()
    df_results['actual_sales'] = y_val
    df_results['predicted_sales'] = predictions

    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    return mae, rmse, df_results, feature_importance_df, model

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("Please upload the `INGRESOS` and `EGRESOS` CSV files.")
    uploaded_ingresos_file = st.file_uploader("Upload INGRESOS", type=['csv'])
    uploaded_egresos_file = st.file_uploader("Upload EGRESOS", type=['csv'])

# --- Main App Interface ---
st.title("📦 Demand Forecasting for Beds & Mattresses")

# Check if files have been uploaded before proceeding
if uploaded_ingresos_file is not None and uploaded_egresos_file is not None:
    # Load and process data once, making it available to all tabs
    df_ingresos, df_egresos = load_data(uploaded_ingresos_file, uploaded_egresos_file)
    df_demand = df_egresos[
        (df_egresos['TIPO_DESC'] == 'Egreso') &
        (df_egresos['ESTADO_DESCRIP'] == 'Cerrada')
    ].copy()

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Exploratory Data Analysis",
        "⚙️ Model Training & Evaluation",
        "📈 Forecast (Coming Soon)"
    ])

    # --- Tab 1: Exploratory Data Analysis ---
    with tab1:
        st.header("Exploratory Data Analysis")
        st.subheader("Data Previews")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**INGRESOS (Inflows):** `{df_ingresos.shape[0]}` rows, `{df_ingresos.shape[1]}` columns.")
            st.dataframe(df_ingresos.head())
        with col2:
            st.write(f"**EGRESOS (Outflows):** `{df_egresos.shape[0]}` rows, `{df_egresos.shape[1]}` columns.")
            st.dataframe(df_egresos.head())

        # Sidebar Filters
        st.sidebar.divider()
        st.sidebar.header("2. Dashboard Filters")
        all_bodegas = sorted(df_demand['BODEGA_ORIGEN_DESC'].unique())
        selected_bodegas = st.sidebar.multiselect('Select Warehouse(s)', options=all_bodegas, default=all_bodegas)

        if not selected_bodegas:
            st.warning("Please select at least one warehouse.")
            st.stop()
        filtered_demand = df_demand[df_demand['BODEGA_ORIGEN_DESC'].isin(selected_bodegas)]

        # KPIs
        st.subheader("High-Level Metrics")
        # (KPI logic remains the same as before...)
        total_quantity_sold = filtered_demand['CANTIDAD'].sum()
        num_unique_skus = filtered_demand['SKU_ALTERNO'].nunique()
        start_date = filtered_demand['GENERADA_EL_FECHA_HORA'].min().date()
        end_date = filtered_demand['GENERADA_EL_FECHA_HORA'].max().date()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Quantity Sold", f"{int(total_quantity_sold):,}")
        kpi2.metric("Unique SKUs Sold", f"{num_unique_skus:,}")
        kpi3.metric("Date Range", f"{start_date} to {end_date}")

        # Visualizations
        st.subheader("Demand Analysis Visualizations")
        sns.set_style("whitegrid")
        # (Plotting logic remains the same as before...)
        weekly_demand = filtered_demand.set_index('GENERADA_EL_FECHA_HORA').resample('W')['CANTIDAD'].sum()
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=weekly_demand, ax=ax1, lw=2)
        ax1.set_title("Total Weekly Demand", fontsize=16)
        st.pyplot(fig1)

        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            top_10_skus = filtered_demand.groupby('SKU_ALTERNO')['CANTIDAD'].sum().nlargest(10)
            fig2, ax2 = plt.subplots(figsize=(8, 6)); sns.barplot(y=top_10_skus.index, x=top_10_skus.values, ax=ax2, palette="viridis"); ax2.set_title("Top 10 SKUs"); st.pyplot(fig2)
        with col_viz2:
            top_10_brands = filtered_demand.groupby('DESC_MARCA')['CANTIDAD'].sum().nlargest(10)
            fig3, ax3 = plt.subplots(figsize=(8, 6)); sns.barplot(y=top_10_brands.index, x=top_10_brands.values, ax=ax3, palette="plasma"); ax3.set_title("Top 10 Brands"); st.pyplot(fig3)

    # --- Tab 2: Model Training & Evaluation ---
    with tab2:
        st.header("Model Training and Evaluation")
        st.markdown("""
        This section allows you to train the demand forecasting model and review its performance on a held-out validation set (the last 12 weeks of data).
        The model uses LightGBM, a powerful gradient boosting framework.
        """)

        # Button to trigger the model training process
        if st.button("🚀 Run Model Training & Evaluation", key='train_model'):
            with st.spinner("Processing data and training model... This may take a few minutes."):
                # 1. Feature Engineering
                df_model_data = create_features(df_demand)

                # 2. Model Training and Evaluation
                mae, rmse, df_results, feature_importance_df, model = train_and_evaluate_model(df_model_data)

                st.success("Model training and evaluation complete!")

                # Display Performance Metrics
                st.subheader("Model Performance Metrics")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                metric_col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                st.info(f"💡 On average, the model's weekly forecast for a single SKU is off by approximately **{mae:.2f} units**.")

                # Visualize Actual vs. Predicted Sales
                st.subheader("Validation Period: Actual vs. Predicted Sales")
                actuals_agg = df_results.groupby('fecha')['actual_sales'].sum()
                predictions_agg = df_results.groupby('fecha')['predicted_sales'].sum()

                fig_val, ax_val = plt.subplots(figsize=(12, 6))
                ax_val.plot(actuals_agg.index, actuals_agg.values, label='Actual Sales', marker='o', linestyle='-')
                ax_val.plot(predictions_agg.index, predictions_agg.values, label='Predicted Sales', marker='x', linestyle='--')
                ax_val.set_title('Total Weekly Sales: Actual vs. Predicted (Validation Set)')
                ax_val.set_ylabel('Total Quantity Sold')
                ax_val.legend()
                st.pyplot(fig_val)

                # Display Feature Importance
                st.subheader("Model Feature Importance")
                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis', ax=ax_imp)
                ax_imp.set_title('Top 15 Most Important Features')
                st.pyplot(fig_imp)
                with st.expander("Why is Feature Importance useful?"):
                    st.markdown("""
                    Feature importance tells us which pieces of information the model found most useful for making predictions.
                    - **Lag features** (e.g., `lag_1`, `lag_4`) being important means that recent sales are a strong predictor of future sales.
                    - **Time features** (e.g., `week_of_year`, `month`) being important indicates seasonality.
                    - **Hierarchy features** (e.g., `DESC_MARCA`, `FAMILIA`) being important shows that the model learns different patterns for different types of products.
                    """)

                # Show a sample of predictions
                st.subheader("Sample of Predictions")
                st.dataframe(df_results.head(10))

    # --- Tab 3: Placeholder for the next module ---
    with tab3:
        st.header("Generate Future Forecast")
        st.info("This section is under construction. After training the model, you will be able to generate and view forecasts for future weeks here.")

else:
    # Initial landing page instructions
    st.markdown("""
    This application provides a comprehensive analysis and forecast of product demand.
    **To begin, please upload your data files using the sidebar on the left.**
    """)
    st.info("Awaiting CSV file uploads...")
