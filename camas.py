import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
# Use the full screen width for a better dashboard experience
st.set_page_config(
    page_title="Demand Forecasting App",
    layout="wide"
)

# --- Data Loading Function ---
# Cache the data loading to improve performance. The data will only be reloaded
# if the uploaded file's contents change.
@st.cache_data
def load_data(ingresos_upload, egresos_upload):
    """
    Loads the inflow and outflow datasets from user-uploaded files,
    parsing date columns for proper handling.
    """
    try:
        # Load the Inflows (Receipts) dataset
        df_ingresos = pd.read_csv(
            ingresos_upload,
            parse_dates=['FechaEmision', 'RecibidaEl', 'TaxDate']
        )

        # Load the Outflows (Shipments/Demand) dataset
        df_egresos = pd.read_csv(
            egresos_upload,
            parse_dates=['GENERADA_EL_FECHA_HORA', 'FECHA_ENTREGA']
        )
        return df_ingresos, df_egresos
    except Exception as e:
        # If loading fails, show a user-friendly error in the app
        st.error(f"Error loading data: {e}")
        st.error("Please ensure the uploaded files are the correct CSVs with the expected columns.")
        return None, None

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("""
    Please upload the two required CSV files:
    - **Ingresos:** `INGRESO_INV_CAMASYCOLCHONES.CSV`
    - **Egresos:** `EGRESO_INV_BODEGA_CAMASYCOLCHONES.CSV`
    """)

    # Create two file uploader widgets
    uploaded_ingresos_file = st.file_uploader(
        "Upload the INGRESOS file",
        type=['csv']
    )
    uploaded_egresos_file = st.file_uploader(
        "Upload the EGRESOS file",
        type=['csv']
    )

# --- Main App Interface ---
st.title("📦 Demand Forecasting for Beds & Mattresses")
st.markdown("""
This application provides a comprehensive analysis and forecast of product demand.
**To begin, please upload your data files using the sidebar on the left.**
""")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "📈 Model & Forecast (Coming Soon)"])

# --- Module 1: Exploratory Data Analysis ---
with tab1:
    st.header("Exploratory Data Analysis")

    # Check if both files have been uploaded before proceeding
    if uploaded_ingresos_file is not None and uploaded_egresos_file is not None:
        # Load data and show a spinner while loading
        with st.spinner('Processing uploaded data... Please wait.'):
            df_ingresos, df_egresos = load_data(uploaded_ingresos_file, uploaded_egresos_file)

        # Check if data was loaded successfully before proceeding
        if df_ingresos is not None and df_egresos is not None:
            st.success("Data loaded successfully!")

            # --- Display Raw Data Previews ---
            st.subheader("Data Previews")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**INGRESOS (Inflows):** `{df_ingresos.shape[0]}` rows, `{df_ingresos.shape[1]}` columns.")
                st.dataframe(df_ingresos.head())
            with col2:
                st.write(f"**EGRESOS (Outflows):** `{df_egresos.shape[0]}` rows, `{df_egresos.shape[1]}` columns.")
                st.dataframe(df_egresos.head())

            # --- Data Pre-processing for Analysis ---
            df_demand = df_egresos[
                (df_egresos['TIPO_DESC'] == 'Egreso') &
                (df_egresos['ESTADO_DESCRIP'] == 'Cerrada')
            ].copy()

            # --- Sidebar Filters (appear after data is loaded) ---
            st.sidebar.divider()
            st.sidebar.header("2. Dashboard Filters")
            st.sidebar.markdown("Use these filters to explore the data.")

            all_bodegas = sorted(df_demand['BODEGA_ORIGEN_DESC'].unique())
            selected_bodegas = st.sidebar.multiselect(
                'Select Warehouse(s)',
                options=all_bodegas,
                default=all_bodegas
            )

            if not selected_bodegas:
                st.warning("Please select at least one warehouse to see the analysis.")
                st.stop()

            filtered_demand = df_demand[df_demand['BODEGA_ORIGEN_DESC'].isin(selected_bodegas)]

            # --- Display Key Performance Indicators (KPIs) ---
            st.subheader("High-Level Metrics")
            st.markdown(f"Metrics below are based on the selected warehouse(s): **{', '.join(selected_bodegas)}**")

            total_quantity_sold = filtered_demand['CANTIDAD'].sum()
            num_unique_skus = filtered_demand['SKU_ALTERNO'].nunique()
            num_transactions = len(filtered_demand)
            start_date = filtered_demand['GENERADA_EL_FECHA_HORA'].min().date()
            end_date = filtered_demand['GENERADA_EL_FECHA_HORA'].max().date()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Quantity Sold", f"{int(total_quantity_sold):,}")
            kpi2.metric("Unique SKUs Sold", f"{num_unique_skus:,}")
            kpi3.metric("Total Transactions", f"{num_transactions:,}")
            kpi4.metric("Date Range", f"{start_date} to {end_date}")

            # --- Data Visualizations ---
            st.subheader("Demand Analysis Visualizations")
            sns.set_style("whitegrid")

            # 1. Total Weekly Demand Plot
            st.markdown("#### Total Weekly Demand Over Time")
            weekly_demand = filtered_demand.set_index('GENERADA_EL_FECHA_HORA').resample('W')['CANTIDAD'].sum()
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=weekly_demand, ax=ax1, lw=2)
            ax1.set_title("Total Weekly Demand", fontsize=16)
            ax1.set_ylabel("Total Quantity Sold")
            ax1.set_xlabel("Date")
            st.pyplot(fig1)

            # 2. Top N Charts
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("#### Top 10 Selling Products (SKU)")
                top_10_skus = filtered_demand.groupby('SKU_ALTERNO')['CANTIDAD'].sum().nlargest(10)
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(y=top_10_skus.index, x=top_10_skus.values, ax=ax2, palette="viridis")
                ax2.set_title("Top 10 SKUs by Quantity Sold")
                ax2.set_xlabel("Total Quantity Sold")
                ax2.set_ylabel("SKU Alterno")
                st.pyplot(fig2)

            with col_viz2:
                st.markdown("#### Top 10 Selling Brands")
                top_10_brands = filtered_demand.groupby('DESC_MARCA')['CANTIDAD'].sum().nlargest(10)
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.barplot(y=top_10_brands.index, x=top_10_brands.values, ax=ax3, palette="plasma")
                ax3.set_title("Top 10 Brands by Quantity Sold")
                ax3.set_xlabel("Total Quantity Sold")
                ax3.set_ylabel("Brand")
                st.pyplot(fig3)

    else:
        # Show a prompt if files are not yet uploaded
        st.info("Awaiting CSV file uploads. Please use the sidebar to upload your data.")
        st.image("https://i.imgur.com/3Z6kH2g.png", width=200) # Simple arrow pointing left

# Placeholder for the next module
with tab2:
    st.header("Model Training and Forecasting")
    st.info("This section is under construction. Please upload data first to enable forecasting.")
