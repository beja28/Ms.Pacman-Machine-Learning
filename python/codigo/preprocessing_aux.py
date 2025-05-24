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
import time

"""
Es utilizada para el modelo de Tabnet
"""

directorio_actual = os.path.dirname(os.path.abspath(__file__))
path_trained = os.path.join(directorio_actual, 'Redes_Entrenadas')

# Columnas
columns_not_scaled = ['ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove', 'pacmanLastMoveMade', 'pacmanMode', 'powerPill_0', 'powerPill_1', 'powerPill_2', 'powerPill_3'] + ['pill_' + str(i) for i in range(220)]
columns_to_categorize = ['ghost1LastMove', 'ghost2LastMove', 'ghost3LastMove', 'ghost4LastMove', 'pacmanLastMoveMade']


# Función de preprocesamiento del CSV con mapeo de 'PacmanMove' a números
def preprocess_csv_aux(path):
    df = pd.read_csv(path)
    
    # Mapeo de la columna 'PacmanMove' a números
    move_mapping = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3
    }
    
    # Aplicamos el mapeo
    df['PacmanMove'] = df['PacmanMove'].map(move_mapping)
        
    # Dividimos el dataframe por intersecciones
    grouped_df = df.groupby(['pacmanCurrentNodeIndex'])

    return grouped_df

# Preprocesar el estado del juego
def preprocess_game_state_aux(game_state, path):
    start_time = time.perf_counter()
    # Leer columnas del CSV (sin columnas derivadas)
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    # Quitar la primera columna (normalmente 'PacmanMove') y la ultima (reward)
    all_cols.remove('PacmanMove')
    all_cols.remove('reward')
    # Quitar también las columnas derivadas
    cols_to_exclude = ["ghostEdibleNearby", "ghostDangerousNearby", "pacmanMode", "closestGhostDistance", "edibleClosestGhostDistance", 
                        "numEdibleGhosts", "ghostsNearby", "numActiveGhosts", "avgGhostDistance", "stdGhostDistance", "minGhostDistance", 
                        "maxGhostDistance","powerPill_0", "powerPill_1", "powerPill_2", "powerPill_3"] + [f'pill_{i}' for i in range(220)]
    
    columns_csv = [col for col in all_cols if col not in cols_to_exclude]

    # Convertir el string de entrada a lista
    data = game_state.strip().split(',')

    totalTime = int(data[0])
    score = int(data[1])
    # Sacar powerPillsState de la posición 21 y convertirlo
    pill_state_str = data[21]
    power_pill_str = data[22]  # No lo quitamos aún, solo lo separamos srea 22

    if not power_pill_str.isdigit() or len(power_pill_str) != 4:
        raise ValueError(f"Valor inválido en powerPillsState: {power_pill_str}")
    
    if len(pill_state_str) != 220 or not all(c in '01' for c in pill_state_str):
        raise ValueError(f"Valor inválido en pillsState: {pill_state_str}")

    data.pop(22)
    data.pop(21)
    data.pop(1)
    data.pop(0)

    if len(columns_csv) != len(data):
        raise ValueError("Distinto número de datos que de columnas")

    # Crear diccionario y convertir tipos
    data_dict = {columns_csv[i]: data[i] for i in range(len(columns_csv))}

    # Convertir el diccionario a DataFrame y guardarlo en CSV (modo añadir)
    pd.DataFrame([data_dict]).to_csv("raw_game_states.csv", mode='a', index=False, header=not os.path.isfile("raw_game_states.csv"))

    # Crear DataFrame base
    df = pd.DataFrame([data_dict])

    # Añadir powerPill
    power_pills_dict = {f"powerPill_{i}": int(power_pill_str[i]) for i in range(4)}

    # Añadir pills
    pills_dict = {f"pill_{i}": int(pill_state_str[i]) for i in range(220)}

    # Convertir columnas necesarias a numéricas
    distance_cols = [f"ghost{i}Distance" for i in range(1, 5)]
    edible_cols = [f"ghost{i}EdibleTime" for i in range(1, 5)]
    lair_cols = [f"ghost{i}LairTime" for i in range(1, 5)]

    df[distance_cols + edible_cols + lair_cols] = df[distance_cols + edible_cols + lair_cols].apply(pd.to_numeric, errors='coerce')

    # Crear matrices
    distances = df[distance_cols].values.astype(float)
    edible_times = df[edible_cols].values.astype(float)
    lair_times = df[lair_cols].values.astype(float)

    distance_threshold = 40
    time_threshold = 5

    active_mask = (distances != -1) & (lair_times == 0)
    masked_distances = np.where(active_mask, distances, -1)
    edible_nearby_mask = (distances < distance_threshold) & (edible_times > time_threshold) & active_mask
    dangerous_nearby_mask = (distances < distance_threshold) & (edible_times == 0) & active_mask

    # Calculamos todas las nuevas features de golpe
    with np.errstate(all='ignore'):
        new_features = {
            "ghostEdibleNearby": edible_nearby_mask.sum(axis=1),
            "ghostDangerousNearby": dangerous_nearby_mask.sum(axis=1),
            "pacmanMode": np.select(
                [
                    (edible_nearby_mask.sum(axis=1) == 0) & (dangerous_nearby_mask.sum(axis=1) == 0),
                    (dangerous_nearby_mask.sum(axis=1) > edible_nearby_mask.sum(axis=1)),
                    (edible_nearby_mask.sum(axis=1) >= dangerous_nearby_mask.sum(axis=1)),
                ],
                [0, 1, 2]
            ),
            "closestGhostDistance": np.nanmin(np.where(active_mask, distances, -1), axis=1),
            "edibleClosestGhostDistance": np.where(
                np.isfinite(np.min(np.where((edible_times > 0) & active_mask, distances, np.inf), axis=1)),
                np.min(np.where((edible_times > 0) & active_mask, distances, np.inf), axis=1),
                -1
            ),
            "numEdibleGhosts": (edible_times > 0).sum(axis=1),
            "ghostsNearby": ((distances < 20) & active_mask).sum(axis=1),
            "numActiveGhosts": (lair_times == 0).sum(axis=1),
            "avgGhostDistance": np.nanmean(masked_distances, axis=1),
            "stdGhostDistance": np.nanstd(masked_distances, axis=1),
            "minGhostDistance": np.nanmin(masked_distances, axis=1),
            "maxGhostDistance": np.nanmax(masked_distances, axis=1),
        }

    # Construimos el nuevo DataFrame con todas las features
    new_cols_df = pd.DataFrame([{**power_pills_dict, **pills_dict, **new_features}])

    # Concatenamos todo de golpe
    df = pd.concat([df.reset_index(drop=True), new_cols_df.reset_index(drop=True)], axis=1)

    # Corregir pacmanMode si ha quedado como array
    if isinstance(df.at[0, "pacmanMode"], (list, np.ndarray)):
        df.at[0, "pacmanMode"] = int(df.at[0, "pacmanMode"][0])

    # Rellenar NaN que pueda quedar por no haber fantasmas activos
    df[["avgGhostDistance", "stdGhostDistance", "minGhostDistance", "maxGhostDistance"]] = df[[
        "avgGhostDistance", "stdGhostDistance", "minGhostDistance", "maxGhostDistance"
    ]].fillna(-1)

    intersection_id = int(df['pacmanCurrentNodeIndex'].iloc[0])
    df = df.drop(columns=['pacmanCurrentNodeIndex'])
    
    # Mapeo de movimientos a números
    move_mapping = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'NEUTRAL': 4
    }

    # Aplicar el mapeo en las columnas categóricas
    for col in columns_to_categorize:
        if col in df.columns:
            df[col] = df[col].map(move_mapping)
    
    scaler_path = os.path.join(path_trained, "models_2025-04-26", f"scaler_({intersection_id},).pkl")
    scaler_bundle = joblib.load(scaler_path)
    scaler = scaler_bundle['scaler']
    columns_scaled = scaler_bundle['columns']
    X_num = df[columns_scaled].fillna(-1)
    X_num_scaled = scaler.transform(X_num)
    X_rest = df[columns_not_scaled]
 
    X_processed = np.hstack([X_num_scaled, X_rest.values])
    df_final = pd.DataFrame(X_processed, columns=columns_scaled + list(X_rest.columns))

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)* 1000

    return df_final, intersection_id
