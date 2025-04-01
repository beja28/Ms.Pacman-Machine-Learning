import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import os
import json
import matplotlib.pyplot as plt
import joblib

directorio_actual = os.path.dirname(os.path.abspath(__file__))
path_trained = os.path.join(directorio_actual, 'Redes_Entrenadas')

# Columnas
columns_to_encode = ['ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove', 'pacmanLastMoveMade']
boolean_col = [
    'pacmanWasEaten', 'ghost1Eaten', 'ghost2Eaten', 'ghost3Eaten', 'ghost4Eaten', 'pillWasEaten', 'isDanger'
]
categories = {
    col: ["DOWN", "LEFT", "NEUTRAL", "RIGHT", "UP"] if col.lower().startswith("ghost")
    else ["DOWN", "LEFT", "RIGHT", "UP"]
    for col in columns_to_encode
}

def preprocess_game_state_aux(game_state, path):
    # Leer columnas del CSV (sin columnas derivadas)
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    # Quitar la primera columna (normalmente 'PacmanMove')
    all_cols = all_cols[1:]
    # Quitar también las columnas derivadas
    cols_to_exclude = ["minGhostDistance", "maxGhostDistance", "meanGhostDistance", "isDanger", "powerPill_0", "powerPill_1", "powerPill_2", "powerPill_3"]
    columns_csv = [col for col in all_cols if col not in cols_to_exclude]

    # Convertir el string de entrada a lista
    data = game_state.strip().split(',')

    # Sacar powerPillsState de la posición 21 y convertirlo
    power_pill_str = data[21]  # No lo quitamos aún, solo lo separamos
    if not power_pill_str.isdigit() or len(power_pill_str) != 4:
        raise ValueError(f"❌ Valor inválido en powerPillsState: {power_pill_str}")


    # Quitar powerPillsState del string de entrada
    data.pop(21)

    if len(columns_csv) != len(data):
        raise ValueError("❌ Distinto número de datos que de columnas")

    # Crear diccionario y convertir tipos
    data_dict = {columns_csv[i]: data[i] for i in range(len(columns_csv))}
    data_dict = convert_types(data_dict)

    # Convertir el diccionario a DataFrame y guardarlo en CSV (modo añadir)
    pd.DataFrame([data_dict]).to_csv("raw_game_states.csv", mode='a', index=False, header=not os.path.isfile("raw_game_states.csv"))

    # Crear DataFrame base
    df = pd.DataFrame([data_dict])

    for i in range(4):
        df[f"powerPill_{i}"] = int(power_pill_str[i])

    # --- Calcular columnas derivadas ---
    distance_cols = ["ghost1Distance", "ghost2Distance", "ghost3Distance", "ghost4Distance"]

    for i in range(1, 5):
        df[f"ghost{i}Active"] = (df[f"ghost{i}Distance"] != -1).astype(int)

    activity_mask = pd.DataFrame({
        col: df[f"ghost{i}Active"] for i, col in enumerate(distance_cols, 1)
    })

    active_distances = df[distance_cols].where(activity_mask == 1, np.nan)

    df["minGhostDistance"] = active_distances.min(axis=1).fillna(-1).round(2)
    df["maxGhostDistance"] = active_distances.max(axis=1).fillna(-1).round(2)
    df["meanGhostDistance"] = active_distances.mean(axis=1).fillna(-1).round(2)

    df["isDanger"] = ((df["minGhostDistance"] >= 0) & (df["minGhostDistance"] < 10)).astype(int)

    for i in range(1, 5):
        del df[f"ghost{i}Active"]

    # One-hot encoding
    encoder = OneHotEncoder(categories=[categories[col] for col in columns_to_encode], sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))

    # Cast de booleanos
    df[boolean_col] = df[boolean_col].astype(int)

    # Eliminar columnas originales y unir codificadas
    df = df.drop(columns=columns_to_encode)
    df = pd.concat([df, encoded_df], axis=1)

    # --- Escalado selectivo ---
    one_hot_cols = [col for col in df.columns if any(prefix in col for prefix in columns_to_encode)]
    columns_no_scale = boolean_col + one_hot_cols

    original_node_index = df['pacmanCurrentNodeIndex'].iloc[0]
    X_num = df.drop(columns=columns_no_scale)
    X_rest = df[columns_no_scale]

    intersection_id = int(df['pacmanCurrentNodeIndex'].iloc[0])
    scaler_path = os.path.join(path_trained, "models_2025-03-29", f"scaler_({intersection_id},).pkl")
    scaler_bundle = joblib.load(scaler_path)
    scaler = scaler_bundle['scaler']
    columns_scaled = scaler_bundle['columns']

    X_num = X_num[columns_scaled]
    X_num_scaled = scaler.transform(X_num)
    X_processed = np.hstack([X_num_scaled, X_rest.values])
    df_final = pd.DataFrame(X_processed, columns=columns_scaled + list(X_rest.columns))

    df_final['pacmanCurrentNodeIndex'] = original_node_index

    file_exists = os.path.isfile("df_final.csv")
    df_final.to_csv("df_final.csv", index=False, mode='a', header=not file_exists)

    return df_final


def convert_types(data_dict):
    for key in boolean_col:
        if key in data_dict:
            val = str(data_dict[key]).strip().lower()
            if val == 'true':
                data_dict[key] = 1
            elif val == 'false':
                data_dict[key] = 0
            else:
                data_dict[key] = int(val)  # Por si acaso llega como 0 o 1 en string

    for key in data_dict:
        if key not in boolean_col and key not in columns_to_encode:
            data_dict[key] = int(data_dict[key])

    return data_dict
