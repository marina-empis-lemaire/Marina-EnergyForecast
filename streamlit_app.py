import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================
# Page Config & Title
# ====================================================
st.set_page_config(page_title="Hybrid Energy Prediction", layout="wide")

st.title("⚡ Hybrid Energy Prediction Model")
st.markdown("""
This app compares a **Theoretical Linear Model** vs. a **Hybrid Model (Linear + XGBoost)** for energy consumption prediction based on Temperature and Time.
""")

# ====================================================
# Sidebar: Settings & Data Upload
# ====================================================
st.sidebar.header("1. Data Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Parameters for the model
st.sidebar.header("2. Model Parameters")
split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.8, 0.05)

st.sidebar.subheader("XGBoost Hyperparameters")
n_estimators = st.sidebar.number_input("N Estimators", 50, 1000, 400, 50)
learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 0.5, 0.05, 0.01)
max_depth = st.sidebar.slider("Max Depth", 1, 10, 4)

# ====================================================
# Main Logic
# ====================================================

if uploaded_file is not None:
    # 1) Load Data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        with st.expander("Raw Data Preview"):
            st.dataframe(df.head())

        # --- Column Selection (Interactive) ---
        cols = df.columns.tolist()
        
        # Attempt to auto-detect columns based on your logic, but let user override
        default_date = next((c for c in cols if 'fecha' in c.lower() or 'date' in c.lower()), cols[0])
        default_energy = next((c for c in cols if 'energ' in c.lower()), cols[0])
        default_temp = next((c for c in cols if 't' in c.lower() or 'temp' in c.lower()), cols[0])

        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Select Date Column", cols, index=cols.index(default_date))
        energy_col = c2.selectbox("Select Energy Column", cols, index=cols.index(default_energy))
        temp_col = c3.selectbox("Select Temperature Column", cols, index=cols.index(default_temp))

        # 2) Preprocessing
        with st.spinner("Preprocessing data..."):
            # Copy to avoid SettingWithCopy warnings
            data = df.copy()
            
            # Timestamp
            data['timestamp'] = pd.to_datetime(data[date_col])
            
            # Standardize columns
            data['energie'] = data[energy_col]
            data['T'] = data[temp_col]
            
            # Sort
            data = data.sort_values('timestamp').reset_index(drop=True)

            # Feature Engineering
            data['hour'] = data['timestamp'].dt.hour
            data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['T2'] = data['T'] ** 2
            data['E_lag'] = data['energie'].shift(1)

            # Drop NA
            data = data.dropna().reset_index(drop=True)
        
        st.write(f"**Data ready:** {data.shape[0]} rows after preprocessing.")

        # ====================================================
        # 3) Theoretical Model (Linear)
        # ====================================================
        st.divider()
        st.header("📊 Model Results")

        base_cols = ['T', 'T2', 'sin_hour', 'cos_hour', 'E_lag']
        X_lin = data[base_cols].values
        y = data['energie'].values

        split_idx = int(split_ratio * len(data))
        X_lin_train, X_lin_test = X_lin[:split_idx], X_lin[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        lin_model = LinearRegression()
        lin_model.fit(X_lin_train, y_train)

        E_theo_train = lin_model.predict(X_lin_train)
        E_theo_test  = lin_model.predict(X_lin_test)

        # Metrics
        rmse_test_theo  = np.sqrt(mean_squared_error(y_test,  E_theo_test))
        r2_test_theo = r2_score(y_test,  E_theo_test)

        # --- Formule ---
        coefs = lin_model.coef_
        intercept = lin_model.intercept_
        
        st.subheader("1. Theoretical Linear Model")
        st.latex(rf"""
        E_{{th}} = {intercept:.3f} + {coefs[0]:.3f} T + {coefs[1]:.3f} T^2 + 
        {coefs[2]:.3f} \sin\left(\frac{{2\pi h}}{{24}}\right) + 
        {coefs[3]:.3f} \cos\left(\frac{{2\pi h}}{{24}}\right) + 
        {coefs[4]:.3f} E_{{lag}}
        """)

        # Save theoretical predictions to dataframe
        data.loc[:split_idx-1, 'E_theo'] = E_theo_train
        data.loc[split_idx:,   'E_theo'] = E_theo_test

        # ====================================================
        # 4) Residuals & XGBoost
        # ====================================================
        data['delta'] = data['energie'] - data['E_theo']

        X_boost = data[base_cols].values
        y_boost = data['delta'].values
        X_boost_train, X_boost_test = X_boost[:split_idx], X_boost[split_idx:]
        y_boost_train, y_boost_test = y_boost[:split_idx], y_boost[split_idx:]

        xgb_model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=0
        )
        
        xgb_model.fit(X_boost_train, y_boost_train)

        delta_pred_train = xgb_model.predict(X_boost_train)
        delta_pred_test  = xgb_model.predict(X_boost_test)

        # ====================================================
        # 5) Hybrid Model Calculation
        # ====================================================
        E_hybrid_train = E_theo_train + delta_pred_train
        E_hybrid_test  = E_theo_test  + delta_pred_test

        rmse_test_hyb  = np.sqrt(mean_squared_error(y_test,  E_hybrid_test))
        r2_test_hyb    = r2_score(y_test,  E_hybrid_test)

        st.subheader("2. Performance Comparison (Test Set)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Linear Model")
            st.metric("RMSE", f"{rmse_test_theo:.4f}")
            st.metric("R² Score", f"{r2_test_theo:.4f}")
        
        with col2:
            st.markdown("### 🚀 Hybrid Model (Linear + XGB)")
            st.metric("RMSE", f"{rmse_test_hyb:.4f}", delta=f"{rmse_test_theo - rmse_test_hyb:.4f}", delta_color="inverse")
            st.metric("R² Score", f"{r2_test_hyb:.4f}", delta=f"{r2_test_hyb - r2_test_theo:.4f}")

        # ====================================================
        # 6) Visualizations (Native Streamlit)
        # ====================================================
        st.divider()
        st.header("📈 Visualizations")
        
        # Calculate full hybrid series for plotting
        # (We predict on the whole dataset for visualization)
        data['Hybrid'] = data['E_theo'] + xgb_model.predict(X_boost)
        
        # Prepare DataFrame for plotting (native charts use index for x-axis usually)
        # We select the columns we want to compare
        plot_df = data[['timestamp', 'energie', 'E_theo', 'Hybrid']].copy()
        plot_df = plot_df.rename(columns={
            'energie': 'Empirical (Measured)',
            'E_theo': 'Theoretical (Linear)',
            'Hybrid': 'Hybrid (Theo + XGB)'
        })
        plot_df = plot_df.set_index('timestamp')

        tab1, tab2 = st.tabs(["Full Period View", "Zoomed View"])

        with tab1:
            st.subheader("Empirical vs Theoretical vs Hybrid (Full period)")
            st.line_chart(plot_df)

        with tab2:
            st.subheader("Zoomed View")
            st.markdown("Adjust the step size to zoom in on the data (plotting every Nth point).")
            n_points_pas = st.slider("Step Size (Plot every Nth point)", 1, 1000, 500)
            
            # Slice the DataFrame based on the slider
            # We use slice notation [start::step]
            zoomed_df = plot_df.iloc[0::n_points_pas]
            
            st.line_chart(zoomed_df)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin.")
    
    # Optional: Generate dummy data for demo
    if st.button("Or generate Sample Data to test"):
        dates = pd.date_range(start="2023-01-01", periods=2000, freq="H")
        temp = 15 + 10 * np.sin(np.linspace(0, 100, 2000)) + np.random.normal(0, 2, 2000)
        # Create synthetic energy with some pattern + noise
        energy = 100 + 5 * temp + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 5, 2000)
        
        dummy_df = pd.DataFrame({
            'Fecha': dates,
            'T': temp,
            'Energía activa (kWh)': energy
        })
        
        # Save to a buffer to mimic an uploaded file
        st.write("Sample data generated! Download it below or just upload your own.")
        st.dataframe(dummy_df.head())
        
        csv = dummy_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Sample CSV",
            csv,
            "sample_energy_data.csv",
            "text/csv",
            key='download-csv'
        )
