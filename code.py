import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================
# 1) Chargement des données
# ====================================================

df = pd.read_csv("energy_Via-Ag-36_withTemp.csv")
print("Colonnes disponibles dans le fichier :")
print(df.columns.tolist())

# --- Détection automatique de la colonne d'énergie ---
energy_col = None
for c in df.columns:
    if "energ" in c.lower():   # ex: "Energía activa (kWh)"
        energy_col = c
        break

if energy_col is None:
    raise ValueError("Impossible de trouver la colonne d'énergie (mot 'energ' non trouvé).")

print("Colonne d'énergie utilisée :", energy_col)

# --- Colonne de temps ---
if 'Fecha' in df.columns:
    df['timestamp'] = pd.to_datetime(df['Fecha'])
elif 'date' in df.columns:
    df['timestamp'] = pd.to_datetime(df['date'])
else:
    raise ValueError("Aucune colonne de date trouvée (ni 'Fecha' ni 'date').")

# --- Colonne d'énergie standardisée ---
df['energie'] = df[energy_col]

# --- Colonne de température ---
if 'T' in df.columns:
    df['T'] = df['T']
else:
    temp_col = None
    for c in df.columns:
        if 'temp' in c.lower():
            temp_col = c
            break
    if temp_col is None:
        raise ValueError("Aucune colonne de température trouvée (ex: 'T').")
    df['T'] = df[temp_col]

# Tri temporel
df = df.sort_values('timestamp').reset_index(drop=True)

# ====================================================
# 2) Construction des features
# ====================================================

df['hour'] = df['timestamp'].dt.hour
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['T2'] = df['T'] ** 2
df['E_lag'] = df['energie'].shift(1)

df = df.dropna().reset_index(drop=True)

# ====================================================
# 3) Modèle théorique (régression linéaire)
# ====================================================

base_cols = ['T', 'T2', 'sin_hour', 'cos_hour', 'E_lag']

X_lin = df[base_cols].values
y = df['energie'].values

split_idx = int(0.8 * len(df))
X_lin_train, X_lin_test = X_lin[:split_idx], X_lin[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

lin_model = LinearRegression()
lin_model.fit(X_lin_train, y_train)

E_theo_train = lin_model.predict(X_lin_train)
E_theo_test  = lin_model.predict(X_lin_test)

rmse_train_theo = np.sqrt(mean_squared_error(y_train, E_theo_train))
rmse_test_theo  = np.sqrt(mean_squared_error(y_test,  E_theo_test))

print("\n=== Modèle théorique (linéaire) ===")
print("RMSE train :", rmse_train_theo)
print("RMSE test  :", rmse_test_theo)
print("R² train   :", r2_score(y_train, E_theo_train))
print("R² test    :", r2_score(y_test,  E_theo_test))

# --- Formule finale du modèle théorique ---
coefs = lin_model.coef_
intercept = lin_model.intercept_

formule = (
    f"E_th = {intercept:.3f}"
    f" + ({coefs[0]:.3f})*T"
    f" + ({coefs[1]:.3f})*T²"
    f" + ({coefs[2]:.3f})*sin(2πh/24)"
    f" + ({coefs[3]:.3f})*cos(2πh/24)"
    f" + ({coefs[4]:.3f})*E_lag"
)

print("\n=== Formule théorique ajustée ===")
print(formule)

df.loc[:split_idx-1, 'E_theo'] = E_theo_train
df.loc[split_idx:,   'E_theo'] = E_theo_test

# ====================================================
# 4) Résidus à corriger par XGBoost
# ====================================================

df['delta'] = df['energie'] - df['E_theo']

X_boost = df[base_cols].values
y_boost = df['delta'].values

X_boost_train, X_boost_test = X_boost[:split_idx], X_boost[split_idx:]
y_boost_train, y_boost_test = y_boost[:split_idx], y_boost[split_idx:]

# ====================================================
# 5) XGBoost sur les résidus
# ====================================================

xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=0
)

# ⚠️ Version ancienne de XGBoost → fit simple
xgb_model.fit(X_boost_train, y_boost_train)

delta_pred_train = xgb_model.predict(X_boost_train)
delta_pred_test  = xgb_model.predict(X_boost_test)

rmse_train_delta = np.sqrt(mean_squared_error(y_boost_train, delta_pred_train))
rmse_test_delta  = np.sqrt(mean_squared_error(y_boost_test,  delta_pred_test))

print("\n=== XGBoost sur les résidus delta_t ===")
print("RMSE delta train :", rmse_train_delta)
print("RMSE delta test  :", rmse_test_delta)

# ====================================================
# 6) Modèle hybride = théorique + correction XGBoost
# ====================================================

E_hybrid_train = E_theo_train + delta_pred_train
E_hybrid_test  = E_theo_test  + delta_pred_test

rmse_train_hyb = np.sqrt(mean_squared_error(y_train, E_hybrid_train))
rmse_test_hyb  = np.sqrt(mean_squared_error(y_test,  E_hybrid_test))
ecarttype=np.sqrt(np.nanvar(E_hybrid_train))

print("\n=== Modèle hybride (théorique + XGBoost) ===")
print("RMSE train :", rmse_train_hyb)
print("RMSE test  :", rmse_test_hyb)
print("R² train   :", r2_score(y_train, E_hybrid_train))
print("R² test    :", r2_score(y_test,  E_hybrid_test))
print("ectype",ecarttype)
# ====================================================
# 7) Visualisation
# ====================================================

time_test = df['timestamp'].iloc[split_idx:]

plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['energie'], label='Empirical (mesuré)', linewidth=2, color='black')
plt.plot(df['timestamp'], df['E_theo'], label='Theoretical', alpha=0.8, color='red')
plt.plot(df['timestamp'], df['E_theo'] + xgb_model.predict(X_boost), label='Hybrid (Theoretical + XGBoost)', alpha=0.8, color='blue')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Empirical vs Theorical vs Hybrid (Full period)')
plt.tight_layout()
plt.show()


# ====================================================
# 7) Visualisation - Période complète
# ====================================================

plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['energie'], label='Empirical (mesuré)', linewidth=2, color='black')
plt.plot(df['timestamp'], df['E_theo'], label='Theoretical', linestyle='--', color='red', alpha=0.8)
plt.plot(df['timestamp'], df['E_theo'] + xgb_model.predict(X_boost), 
         label='Hybrid (Theoretical + XGBoost)', color='blue', alpha=0.8)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Empirical vs Theoretical vs Hybrid (Full period)')
plt.tight_layout()
plt.show()

# ====================================================
# 8) Visualisation - Zoom avec un pas
# ====================================================

n_points_pas = 500  # nombre de pas

time_zoom = df['timestamp'].iloc[0::n_points_pas]
E_emp_zoom = df['energie'].iloc[0::n_points_pas]
E_theo_zoom = df['E_theo'].iloc[0::n_points_pas]
E_hyb_zoom = (df['E_theo'] + xgb_model.predict(X_boost)).iloc[0::n_points_pas]

plt.figure(figsize=(12, 5))
plt.plot(time_zoom, E_emp_zoom, label='Empirical (mesuré)', linewidth=2, color='black')
plt.plot(time_zoom, E_theo_zoom, label='Theoretical', linestyle='--', color='red')
plt.plot(time_zoom, E_hyb_zoom, label='Hybrid (Theoretical + XGBoost)', color='blue')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title(f'step for each {n_points_pas} hours')
plt.tight_layout()
plt.show()


