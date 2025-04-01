import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

# Archivos de entrada y salida
input_csv = "18_gameStatesData.csv"
output_csv = "18_gameStatesData_enriched.csv"

# ---------------------------
# 📥 Cargar y filtrar por mazeIndex
# ---------------------------
df = pd.read_csv(
    input_csv,
    low_memory=False,
    on_bad_lines='skip',
    dtype={"powerPillsState": str}
)

print(f"✅ CSV cargado con éxito. Total de filas leídas: {len(df)}")

# Filtrar solo las filas con mazeIndex == 0
df = df[df["mazeIndex"] == 0].reset_index(drop=True)
print(f"🧮 Filas tras filtrar por mazeIndex == 0: {len(df)}")

# -------------------------
# ⚡️ Procesar powerPillsState
# -------------------------
print("🔍 Valores únicos en powerPillsState:")
print(df["powerPillsState"].unique())
# Asegurarse de que son strings de longitud 4 compuestas de 0s y 1s
valid_power_pills = df["powerPillsState"].astype(str).str.extract(r'([01]{4})')[0]

# Filtrar filas válidas
valid_mask = valid_power_pills.notnull()
valid_power_pills = valid_power_pills[valid_mask]

# 🧮 Mostrar cuántas filas fueron eliminadas
print(f"⚠️ Se eliminaron {len(df) - len(valid_power_pills)} filas con powerPillsState inválido.")

# Aplicar máscara a df para mantener solo las filas válidas
df = df.loc[valid_power_pills.index].reset_index(drop=True)
valid_power_pills = valid_power_pills.reset_index(drop=True)

# Crear columnas binarias
power_matrix = np.array(valid_power_pills.apply(list).tolist(), dtype=np.uint8)
power_df = pd.DataFrame(power_matrix, columns=[f'powerPill_{i}' for i in range(4)])

# Añadirlas al DataFrame
df = pd.concat([df, power_df], axis=1)

# Eliminar columna original
df = df.drop(columns=["powerPillsState"], errors="ignore")

# -----------------------------------
# 📊 Enriquecimiento fantasma
# -----------------------------------

# Columnas relacionadas con fantasmas
distance_cols = ["ghost1Distance", "ghost2Distance", "ghost3Distance", "ghost4Distance"]
for col in distance_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Marcar fantasmas activos (solo para calcular distancias, luego se eliminan)
for i in range(1, 5):
    df[f"ghost{i}Active"] = (df[f"ghost{i}Distance"] != -1).astype(int)

# Crear máscara de fantasmas activos
activity_mask = pd.DataFrame({
    col: df[f"ghost{i}Active"] for i, col in enumerate(distance_cols, 1)
})

# Usar solo distancias de fantasmas activos
active_distances = df[distance_cols].where(activity_mask == 1, np.nan)

# Calcular estadísticas sobre fantasmas activos
df["minGhostDistance"] = active_distances.min(axis=1)
df["maxGhostDistance"] = active_distances.max(axis=1)
df["meanGhostDistance"] = active_distances.mean(axis=1)

# Reemplazar NaNs con -1 y redondear a 2 decimales
dist_cols = ["minGhostDistance", "maxGhostDistance", "meanGhostDistance"]
df[dist_cols] = df[dist_cols].fillna(-1).round(2)

# Situación de peligro: ¿algún fantasma activo a menos de 10 unidades?
df["isDanger"] = ((df["minGhostDistance"] >= 0) & (df["minGhostDistance"] < 10)).astype(int)

# Eliminar columnas 'ghostXActive' (ya no necesarias)
for i in range(1, 5):
    del df[f"ghost{i}Active"]

del df["mazeIndex"]

# Guardar CSV final
df.to_csv(output_csv, index=False, float_format="%.2f")
print(f"✅ CSV enriquecido guardado en: {output_csv}")
