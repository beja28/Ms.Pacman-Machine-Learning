import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import json
import torch
from sklearn.inspection import permutation_importance
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from datetime import datetime
import re
from Pytorch_Predictor import PyTorchPredictor


class Explainability:
    def __init__(self):
        self.explanations = []  # Almacena resultados de explicabilidad
        self.map_dict = {}  # Diccionario para almacenar los valores SHAP organizados por intersección

    def run_explainability(self, model, model_filename, technique, X, key, Y=None):
        """Ejecuta la técnica de explicabilidad seleccionada."""
        if technique == "shap":
            self.explain_with_shap(model, model_filename, X, key)
        elif technique == "feature_importance":
            self.explain_with_feature_importance(model, model_filename, X, Y)
        elif technique == "lime":
            self.explain_with_lime(model, model_filename, X)
            
    def explain_with_shap(self, model, model_filename, X, key):
        """Implementación de SHAP para explicar las predicciones del modelo."""
        try:
            print({key})
            global_shap_values = None
            feature_names = X.columns

            if isinstance(model, torch.nn.Module):  # Si es un modelo PyTorch
                background_data = X.sample(n=200)
                data_tensor = torch.tensor(background_data.values, dtype=torch.float32)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                data_tensor = data_tensor.to(device)

                print(f"Ejecuta la explicabilidad para el modelo {model_filename}")

                explainer = shap.DeepExplainer(model, data_tensor)

                data_to_explain = X.sample(n=10)
                data_tensor_to_explain = torch.tensor(data_to_explain.values, dtype=torch.float32).to(device)

                shap_values = explainer.shap_values(data_tensor_to_explain)
                global_shap_values = shap_values
            else:
                background_data = shap.sample(X, 200)
                
                print(f"Ejecuta la explicabilidad para el modelo {model_filename}")
                
                explainer = shap.KernelExplainer(model.predict, background_data)
                
                data_to_explain = X.sample(n=100)
                shap_values = explainer.shap_values(data_to_explain)
                global_shap_values = shap_values
                

            self.explanations.append({
                "technique": "shap",
                "shap_values": global_shap_values,
                "feature_names": feature_names
            })
            # Guardar los valores SHAP en un diccionario estructurado por intersección y característica
            self.map_dict[key] = {feature: value for feature, value in zip(feature_names, shap_values[0])}

        except Exception as e:
            print(f"Error durante la explicabilidad SHAP: {e}")

    def explain_with_feature_importance(self, model, model_filename, X, Y):
        """Implementación de Feature Importance."""
        
        # Scikit-Learn
        if hasattr(model, 'predict'):
            print(f"Ejecuta la explicabilidad para el modelo {model_filename}")
            feature_names = X.columns

            X_sampled = X.sample(n=500, random_state=42)
            Y_sampled = Y[X_sampled.index]

            results = permutation_importance(
                model,
                X_sampled,
                Y_sampled,
                n_repeats=5,
                random_state=1
            )
            importances = results.importances_mean

            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
        else:
            raise ValueError("Modelo no compatible con feature importance")

        feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
        # Guardar los resultados de Feature Importance
        self.explanations.append({
            "technique": "feature_importance",
            "importances": feature_importances,
            "feature_names": feature_names
        })

    def explain_with_lime(self, model, model_filename, X, node_index=165):
        """Implementacion de LIME para explicar las predicciones del modelo en una interseccion en concreto (funciona para Scikit-Learn y PyTorch)."""
        try:

            # Encontramos el nodo de interseccion de ese modelo de red, que viene en el nombre entre ()
            match = re.search(r'\((\d+),\)', model_filename)
            model_node_index = int(match.group(1)) if match else None
            if model_node_index != node_index:
                print(f"--> Saltando modelo {model_filename}, no corresponde al nodo {node_index}")
                return

            # Si X no es un DataFrame, se convierte en uno para manejar las caracteristicas adecuadamente
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=model.feature_names_in_)

            # Filtramos los estados que estan en la posicion deseada
            filtered_states = X[X["pacmanCurrentNodeIndex"] == node_index]
            if filtered_states.empty:
                print(f"\n No se encontraron estados con pacmanCurrentNodeIndex = {node_index} para el modelo {model_filename}. Saltando explicabilidad LIME.\n")
                return

            # Se elige aleatoriamente uno de esos estados
            i = np.random.randint(0, len(filtered_states))
            instance = filtered_states.iloc[i]

            print("\nEstado seleccionado para explicación con LIME:")
            print(instance.to_frame().T)

            # Crea el explicador de LIME
            explainer = LimeTabularExplainer(
                X.values,
                mode="classification",
                feature_names=X.columns,
                discretize_continuous=True
            )
            
            # Si el modelo es de Scikit-Learn, usa predict_proba. Si es de PyTorch, crea un predict_proba personalizado
            if hasattr(model, "predict_proba"):
                # funcion para LIME (obligatoriamente debe ser predict_proba)
                predict_fn = model.predict_proba

                # prediccion real del modelo en este estado
                moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NEUTRAL']
                probabilidades = model.predict_proba(instance.values.reshape(1, -1))[0]
                predicted_index = np.argmax(probabilidades)

                # Convertimos ese índice numérico al nombre del movimiento
                y_pred = moves[predicted_index]

            elif isinstance(model, torch.nn.Module):
                # Para modelos de PyTorch, crea una función predict_proba
                def predict_fn(X_input):
                    model.eval()  # Cambia a modo evaluación
                    with torch.no_grad():  # Desactiva gradientes para eficiencia
                        X_tensor = torch.tensor(X_input, dtype=torch.float32)
                        outputs = model(X_tensor)
                        return torch.softmax(outputs, dim=1).cpu().numpy()  # Devuelve las probabilidades
                    

                # Prediccion real del modelo en el estado seleccionado
                model.eval()
                with torch.no_grad():
                    instance_tensor = torch.tensor(instance.values.reshape(1, -1), dtype=torch.float32)
                    output = model(instance_tensor)
                    y_pred = model.predict(instance.values.reshape(1, -1))[0]
            else:
                raise NotImplementedError("El modelo debe implementar 'predict_proba' o ser de PyTorch.")

            # Explica la instancia seleccionada usando LIME
            print(f"\nEl movimiento predecido por el modelo {model_filename} en la posicion {node_index} ha sido =>> {y_pred}")
            print(f"Ejecutando la explicabilidad LIME para el modelo {model_filename} \n")
            exp = explainer.explain_instance(instance.values, predict_fn, num_features=5)
            
            # Guarda los resultados de la explicacion LIME
            self.explanations.append({
                "technique": "lime", 
                "lime_exp": exp,
                "feature_names": X.columns,
                "modelo_filename": model_filename
            })

            # Guardar en el .txt de explicacioones LIME
            lime_folder = os.path.join('..', 'images', 'Explicabilidad', 'Lime')
            os.makedirs(lime_folder, exist_ok=True)

            # Hora del sistema para identificar y diferenciar
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            lime_txt_path = os.path.join(lime_folder, 'explicaciones_LIME.txt')

            try:
                # Si no existe se crea el txt y la cabecera
                if not os.path.exists(lime_txt_path):
                    with open(lime_txt_path, 'w', encoding='utf-8') as f:
                        f.write('Archivo recopilatorio de explicaciones generadas con LIME\n')
                        f.write('------------------------------------------------------------------\n\n')             

                # Se abre en modo append para meter nueva info
                with open(lime_txt_path, 'a', encoding='utf-8') as f:
                    f.write(f'Explicacion LIME para el modelo: {model_filename}\n')
                    f.write(f'Fecha y hora: {current_time}\n')
                    f.write(f'Movimiento predecido por el modelo para este estado ha sido: {y_pred}\n')
                    f.write('Estado analizado:\n')
                    f.write(instance.to_frame().T.to_string(index=False))
                    f.write('\n\nCaracteristicas mas influyentes:\n')
                    for feature, influence in exp.as_list():
                        f.write(f' - {feature}: {influence:.4f}\n')
                    f.write('\n--------------------------------------------------\n\n')

            except Exception as e:
                print(f"Error guardando explicaciones en .txt: {e}")

            # Se guarda el grafico en la misma carpeta
            self.generate_lime_plot(model_filename, node_index, y_pred)

        except Exception as e:
            print(f"Error durante la explicabilidad LIME: {e}")


    def generate_lime_plot(self, model_filename, node_index, move):
        """ Genera y guarda una grafica LIME para un modelo especifico en su carpeta correspondiente """

        # Busca el LIME correspondiente al modelo
        lime_exps = [exp for exp in self.explanations if exp["technique"] == "lime" and exp.get("modelo_filename") == model_filename]

        if not lime_exps:
            print(f"No se encontraron explicaciones LIME para el modelo: {model_filename}")
            return

        lime_exp = lime_exps[0]["lime_exp"]

        # Crea la figura de LIME
        fig = lime_exp.as_pyplot_figure()
        plt.title(f"Explicabilidad LIME")
        plt.yticks(fontsize=10)
        plt.subplots_adjust(left=0.4)
        plt.xlabel('Valor de impacto', fontsize=10)
        plt.ylabel('Características', fontsize=10)


        # Comentario personalizado debajo del gráfico
        comentario = f"Intersección número: {node_index} --- Movimiento realizado: {move}"
        plt.figtext(0.5, -0.05, comentario, ha='center', fontsize=10)

        # Ruta donde esta la explicabilidad
        lime_folder = os.path.join('..', 'images', 'Explicabilidad', 'Lime')
        os.makedirs(lime_folder, exist_ok=True)

        # El nombre del archivo, es el nombre del modelo y la hora, para diferenciarlos
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"LIME_{model_filename}_{current_time}.png"
        full_path = os.path.join(lime_folder, filename)

        plt.savefig(full_path, bbox_inches='tight')
        plt.close()

        print(f"Grafico de explicabilidad LIME guardado en: {full_path} \n")

    def generate_tabnet_feature_importance(self, path):
        all_importances = []
        feature_names = None

        for file in os.listdir(path):
            if file.startswith("feature_importances"):
                full_path = os.path.join(path, file)
                with open(full_path, 'r') as f:
                    data = json.load(f)

                    if feature_names is None:
                        feature_names = data["features"]
                    else:
                        assert data["features"] == feature_names, f"Features distintas en {file}"

                    all_importances.append(data["importances"])

        importances_df = pd.DataFrame(all_importances, columns=feature_names)
        mean_importances = importances_df.mean().sort_values(ascending=False)

        return mean_importances

    def generate_global_explainability_plot(self, model, technique, path):
        """Genera un gráfico que combine la explicabilidad global de todas las redes."""
        
        # Combinar todos los valores SHAP
        shap_exps = [exp for exp in self.explanations if exp["technique"] == "shap"]
        if technique == "shap" and shap_exps:
            all_shap_values = [exp["shap_values"] for exp in shap_exps]
            all_shap_values = np.concatenate(all_shap_values, axis=0)

            # Calcular la importancia media de cada característica
            mean_shap_values = np.mean(np.abs(all_shap_values), axis=0)

            feature_names = shap_exps[0]["feature_names"]

            if model == "pytorch":
                # Reducir a una dimension
                mean_shap_values = np.mean(mean_shap_values, axis=1)

            #Ordenar por importancia (mayor a menor)
            sorted_idx = np.argsort(mean_shap_values)[::-1]
            sorted_feature_names = np.array(feature_names)[sorted_idx]
            sorted_mean_shap_values = mean_shap_values[sorted_idx]

            plt.figure(figsize=(10, 20))
            plt.barh(sorted_feature_names, sorted_mean_shap_values, color='skyblue')
            plt.gca().invert_yaxis()
            plt.xlabel("Importancia media de SHAP", fontsize=10)
            plt.ylabel("Características", fontsize=10)
            plt.title("Importancia global de características (SHAP)")
            plt.yticks(fontsize=14)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            plt.show()

        elif technique == "feature_importance":
            if model == "tabnet":
                mean_importances = self.generate_tabnet_feature_importance(path)
            else:
                feature_imp_exps = [exp for exp in self.explanations if exp["technique"] == "feature_importance"]
                if feature_imp_exps:
                    importances_list = [exp["importances"] for exp in feature_imp_exps]
                    combined = pd.concat(importances_list, axis=0)
                    mean_importances = combined.groupby('Feature')['Importance'].mean().sort_values(ascending=False)

            plt.figure(figsize=(10, 20))
            mean_importances.plot(kind='barh', color='skyblue')
            plt.gca().invert_yaxis()
            plt.xlabel("Mean Importance")
            plt.ylabel("Features", fontsize=10)
            plt.title("Importancia Global de Características")
            plt.yticks(fontsize=14)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            plt.show()

    def calculate_intersection_impact(self, model, models_dir):
        """Calcula el impacto total de cada característica en cada intersección."""
        intersection_impact = {}

        if model == "tabnet":
            for filename in os.listdir(models_dir):
                if filename.startswith("feature_importances"):
                    intersection = filename.split("_")[-1].replace(".json", "") # Extrae del nombre la intersección como (intersección,)
                    full_path = os.path.join(models_dir, filename)
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        features = data["features"]
                        importances = data["importances"]
                        intersection_impact[intersection] = dict(zip(features, importances))

        for intersection, features in self.map_dict.items():
            intersection_impact[intersection] = {
                feature: np.mean(shap_values)
                for feature, shap_values in features.items()
            }

        return intersection_impact

    def save_explainability_txt(self, directory, models_dir, model):
        """Guarda un archivo TXT para cada característica con el impacto en cada intersección."""
        if model == "pytorch":
            full_path = os.path.join(directory, "mapas_explicabilidad_txt_pytorch")
        elif model == "sklearn":
            full_path = os.path.join(directory, "mapas_explicabilidad_txt_sklearn")
        elif model == "tabnet":
            full_path = os.path.join(directory, "mapas_explicabilidad_txt_tabnet")

        os.makedirs(full_path, exist_ok=True)

        intersection_impact = self.calculate_intersection_impact(model, models_dir)

        features = set()
        for impacts in intersection_impact.values():
            features.update(impacts.keys())

        for feature in features:
            file_path = os.path.join(full_path, f"{feature}.txt")
            with open(file_path, "w") as f:
                for intersection, impacts in intersection_impact.items():
                    if feature in impacts:
                        f.write(f"Intersección {intersection}: {impacts[feature]:.6f}\n")

        print(f"Archivos guardados en: {full_path}")





