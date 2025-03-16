from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import numpy as np
import os

def train_tabnet_nn(X_train, y_train, X_cv, y_cv, max_epochs=150):
    """
    Entrena un modelo TabNet optimizado para GPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 🔥 Detectar GPU
    print(f"🔥 Usando dispositivo: {device}")

    clf = TabNetClassifier(
        device_name=device,  # ✅ Forzar uso de GPU si está disponible
        n_d=16,  
        n_a=16,  
        n_steps=4,  
        gamma=1.3,  
        lambda_sparse=0.005,  
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},  
        scheduler_params={"step_size": 50, "gamma": 0.9},  
        scheduler_fn=torch.optim.lr_scheduler.StepLR
    )

    # 🚀 Entrenamiento con GPU
    clf.fit(
        X_train, y_train,
        eval_set=[(X_cv, y_cv)],
        max_epochs=max_epochs,
        patience=10,  # 🔥 Reducido para evitar entrenamiento innecesario
        batch_size=4096  # 🚀 Aumentamos para aprovechar la GPU
    )

    return clf

def save_model_tabnet(model, save_path, key):
    """
    Guarda un modelo TabNet en un directorio específico.
    """
    model_dir = os.path.join(save_path, "models_tabnet")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"tabnet_model_{key}.zip")
    model.save_model(model_path)
    print(f"✅ Modelo TabNet guardado en: {model_path}")

def load_model_tabnet(model_path):
    """
    Carga un modelo TabNet guardado y lo mueve a la GPU si está disponible.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabNetClassifier(device_name=device)  # 🔥 Asegurar que carga en GPU
    model.load_model(model_path)
    print(f"✅ Modelo TabNet cargado en: {device}")
    return model
