import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from datetime import datetime
import re


class Explicabilidad:
    def __init__(self):
        self.explicaciones = []  # Almacena resultados de explicabilidad
        self.dic_map = {} # Diccionario para almacenar los valores SHAP organizados por interseccion 

    def ejecutar_explicabilidad(self, model, model_filename, technique, X, key, Y=None):
        """Ejecuta la técnica de explicabilidad seleccionada."""
        if technique == "shap":
            self.explicabilidad_shap(model, model_filename, X, key)
        elif technique == "feature_importance":
            self.explicabilidad_feature_importance(model, model_filename, X, Y)
        elif technique == "lime":
            self.explicabilidad_lime(model, model_filename, X)

    
    def explicabilidad_shap(self, model, model_filename, X, key):
        """Implementación de SHAP para explicar las predicciones del modelo."""
        try:
            print({key})
            shap_values_global = None
            feature_names = X.columns  # Guardamos los nombres de las características

            if isinstance(model, torch.nn.Module):  # Si es un modelo PyTorch
                background_data = X.sample(n=200)
                data_tensor = torch.tensor(background_data.values, dtype=torch.float32)
                
                # Configurar dispositivo
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                data_tensor = data_tensor.to(device)
                
                print(f"Ejecuta la explicabilidad para el modelo {model_filename}")
                
                explainer = shap.DeepExplainer(model, data_tensor)
                
                data_to_explain = X.sample(n=10)
                data_to_explain_tensor = torch.tensor(data_to_explain.values, dtype=torch.float32).to(device)
                
                shap_values = explainer.shap_values(data_to_explain_tensor)
                shap_values_global = shap_values
            else:
                background_data = shap.sample(X, 200)
                
                print(f"Ejecuta la explicabilidad para el modelo {model_filename}")

                explainer = shap.KernelExplainer(model.predict, background_data)

                data_to_explain = X.sample(n=100)
                shap_values = explainer.shap_values(data_to_explain)  # Calcula los valores SHAP
                shap_values_global = shap_values 

            # Guardar los resultados de SHAP, incluyendo los nombres de las características
            self.explicaciones.append({
                "technique": "shap", 
                "shap_values": shap_values_global, 
                "feature_names": feature_names
            })
            
            # Guardar los valores SHAP en un diccionario estructurado por intersección y característica
            self.dic_map[key] = {feature: value for feature, value in zip(feature_names, shap_values[0])}
        
        except Exception as e:
            print(f"Error durante la explicabilidad SHAP: {e}")

    def explicabilidad_feature_importance(self, model, model_filename, X, Y):
        """Implementación de Feature Importance utilizando Permutation Importance."""
        try:
            if hasattr(model, 'predict'):  # Comprobar que tiene método predict
                X_sampled = X.sample(n=500, random_state=42)  # Muestra de 500 instancias
                Y_sampled = Y[X_sampled.index]  # Etiquetas correspondientes a las muestras

                print(f"Ejecuta la explicabilidad para el modelo {model_filename}")

                results = permutation_importance(
                    model, 
                    X_sampled, 
                    Y_sampled, 
                    n_repeats=5,  # Reducir repeticiones para mejorar rendimiento
                    random_state=1
                )
                importances = results.importances_mean 

                feature_names = X.columns
                feature_importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })

                feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

                # Guardar los resultados de Feature Importance
                self.explicaciones.append({
                    "technique": "feature_importance", 
                    "importances": feature_importances,
                    "feature_names": feature_names
                })

        except Exception as e:
            print(f"Error durante la explicabilidad Feature Importance: {e}")
    

    def explicabilidad_lime(self, model, model_filename, X, node_index=165):
        """Implementación de LIME para explicar las predicciones del modelo en una interseccion en concreto (funciona para Scikit-Learn y PyTorch)."""
        try:

            match = re.search(r'\((\d+),\)', model_filename)
            modelo_node_index = int(match.group(1)) if match else None

            if modelo_node_index != node_index:
                print(f"⏩ Saltando modelo {model_filename}, no corresponde al nodo {node_index}")
                return

            # Si X no es un DataFrame, se convierte en uno para manejar las características adecuadamente
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=model.feature_names_in_)

            estados_filtrados = X[X["pacmanCurrentNodeIndex"] == node_index]

            if estados_filtrados.empty:
                print(f"\n No se encontraron estados con pacmanCurrentNodeIndex = {node_index} para el modelo {model_filename}. Saltando explicabilidad LIME.\n")
                return

            # Selecciona aleatoriamente un estado filtrado
            i = np.random.randint(0, len(estados_filtrados))
            instance = estados_filtrados.iloc[i]

            print("\n🕹 **Estado seleccionado para explicación con LIME:**")
            print(instance.to_frame().T)  # Transponer para mejor lectura

            # Crea el explicador de LIME
            explainer = LimeTabularExplainer(
                X.values,
                mode="classification",
                feature_names=X.columns,
                discretize_continuous=True
            )
            
            # Si el modelo es de Scikit-Learn, usa predict_proba. Si es de PyTorch, crea un predict_proba personalizado
            if hasattr(model, "predict_proba"):
                # Para modelos de Scikit-Learn, usa predict_proba directamente
                predict_fn = model.predict_proba
            elif isinstance(model, torch.nn.Module):
                # Para modelos de PyTorch, crea una función predict_proba
                def predict_fn(X_input):
                    model.eval()  # Cambia a modo evaluación
                    with torch.no_grad():  # Desactiva gradientes para eficiencia
                        X_tensor = torch.tensor(X_input, dtype=torch.float32)
                        outputs = model(X_tensor)
                        return torch.softmax(outputs, dim=1).cpu().numpy()  # Devuelve las probabilidades
            else:
                raise NotImplementedError("El modelo debe implementar 'predict_proba' o ser de PyTorch.")

            # Explica la instancia seleccionada usando LIME
            print(f"Ejecutando la explicabilidad LIME para el modelo {model_filename}")
            exp = explainer.explain_instance(instance.values, predict_fn, num_features=5)
            
            # Guarda los resultados de la explicación LIME
            self.explicaciones.append({
                "technique": "lime", 
                "lime_exp": exp,
                "feature_names": X.columns,
                "modelo_filename": model_filename
            })

            # Guardar en un .txt
            ruta_carpeta = os.path.join('..', 'images', 'Explicabilidad', 'Lime')
            os.makedirs(ruta_carpeta, exist_ok=True)

            hora_act = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ruta_txt = os.path.join(ruta_carpeta, 'explicaciones_LIME.txt')

            try:
                # Si no existe se crea el txt y la cabecera
                if not os.path.exists(ruta_txt):
                    with open(ruta_txt, 'w', encoding='utf-8') as f:
                        f.write('Archivo recopilatorio de explicaciones generadas con LIME\n')
                        f.write('------------------------------------------------------------------\n\n')             

                # Luego, siempre abre en modo append para meter nueva info
                with open(ruta_txt, 'a', encoding='utf-8') as f:
                    f.write(f'Explicacion LIME para el modelo: {model_filename}\n')
                    f.write(f'Fecha y hora: {hora_act}\n')
                    f.write('Estado analizado:\n')
                    f.write(instance.to_frame().T.to_string(index=False))
                    f.write('\n\nCaracterísticas mas influyentes:\n')
                    for feature, influence in exp.as_list():
                        f.write(f' - {feature}: {influence:.4f}\n')
                    f.write('\n--------------------------------------------------\n\n')

            except Exception as e:
                print(f"Error guardando explicaciones en TXT: {e}")

            # Se guarda el grafico
            self.generar_grafico_lime_individual(model_filename)

        except Exception as e:
            print(f"Error durante la explicabilidad LIME: {e}")


    def generar_grafico_lime_individual(self, model_filename):
        """ Genera y guarda una grafica LIME para un modelo especifico en su carpeta correspondiente """

        # Busca el LIME correspondiente al modelo
        lime_exps = [exp for exp in self.explicaciones if exp["technique"] == "lime" and exp.get("modelo_filename") == model_filename]

        if not lime_exps:
            print(f"No se encontraron explicaciones LIME para el modelo: {model_filename}")
            return

        lime_exp = lime_exps[0]["lime_exp"]

        # Crear la figura de LIME
        fig = lime_exp.as_pyplot_figure()
        plt.title(f"Explicabilidad LIME")

        # Estética del gráfico
        plt.yticks(fontsize=10)
        plt.subplots_adjust(left=0.4)
        plt.xlabel('Valor de impacto', fontsize=10)
        plt.ylabel('Características', fontsize=10)

        # Ruta para guardar
        ruta_carpeta = os.path.join('..', 'images', 'Explicabilidad', 'Lime')
        os.makedirs(ruta_carpeta, exist_ok=True)

        # Nombre del archivo con timestamp
        hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre_archivo = f"LIME_{model_filename}_{hora_actual}.png"
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)

        # Guardar el gráfico
        plt.savefig(ruta_completa, bbox_inches='tight')
        plt.close()

        print(f"Gráfico de explicabilidad LIME guardado en: {ruta_completa}")
        
        
    def generar_grafico_explicabilidad_global(self, model):
        """Genera un gráfico que combine la explicabilidad global de todas las redes y lo guarda en un archivo."""

        # Combinar todos los valores SHAP
        shap_exps = [exp for exp in self.explicaciones if exp["technique"] == "shap"]
        if shap_exps:
            # Concatenar todos los valores SHAP de cada modelo
            all_shap_values = [exp["shap_values"] for exp in shap_exps]
            all_shap_values = np.concatenate(all_shap_values, axis=0)

            # Calcular la importancia media de cada característica (media de los valores SHAP)
            mean_shap_values = np.mean(np.abs(all_shap_values), axis=0)  # Valor absoluto de los SHAP para ver la importancia global

            # Obtener los nombres de las características (usados en todos los modelos)
            feature_names = shap_exps[0]["feature_names"]
            
            if model == "pytorch":                           
                # Reducir `mean_shap_values` a una dimension
                mean_shap_values = np.mean(mean_shap_values, axis=1)              
                      
            # Ordenar por la importancia (de mayor a menor)
            sorted_idx = np.argsort(mean_shap_values)[::-1]

            # Ordenar nombres y valores
            sorted_feature_names = np.array(feature_names)[sorted_idx]
            sorted_mean_shap_values = mean_shap_values[sorted_idx]
            

            # Crear gráfico de barras
            plt.figure(figsize=(10, 20))
            plt.barh(sorted_feature_names, sorted_mean_shap_values, color='skyblue')
            plt.gca().invert_yaxis()  # Invertir el eje Y para que el de mayor importancia quede arriba
            plt.xlabel("Importancia media de SHAP", fontsize=10)
            plt.ylabel("Características", fontsize=10)
            plt.title("Importancia global de características (SHAP)")
            
            # Ajustar el tamaño de las etiquetas del eje Y
            plt.yticks(fontsize=14)  # Tamaño de letra más pequeño para las etiquetas del eje Y

            # Ajustar el espaciado de los nombres de las características
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            
            plt.show()


        # Generar gráfico de Importancia de características global
        feature_importance_exps = [exp for exp in self.explicaciones if exp["technique"] == "feature_importance"]
        if feature_importance_exps:
            importancias_list = []

            # Extraer las importancias de características de todos los modelos
            for explicacion in feature_importance_exps:
                importancias_list.append(explicacion["importances"])

            # Combinar las importancias promediando a través de todos los modelos
            if importancias_list:
                combined_importances = pd.concat(importancias_list, axis=0)
                importancias_mean = combined_importances.groupby('Feature')['Importance'].mean().sort_values(ascending=False)

                # Generar la gráfica combinada de Feature Importance
                plt.figure(figsize=(10, 20))
                importancias_mean.plot(kind='barh', color='skyblue')
                plt.gca().invert_yaxis()
                plt.xlabel("Mean Importance")
                plt.ylabel("Features", fontsize=10)
                plt.title("Importancia Global de Características")

                # Ajustar el tamaño de las etiquetas del eje Y
                plt.yticks(fontsize=14)  # Tamaño de letra más pequeño para las etiquetas del eje Y

                # Ajustar el espaciado de los nombres de las características
                plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                
                plt.show()

        # Generar gráfico LIME global
        lime_exps = [exp for exp in self.explicaciones if exp["technique"] == "lime"]
        if lime_exps:
            # Usar el primer experimento de LIME, aunque podrías combinar los resultados si lo deseas
            lime_exp = lime_exps[0]["lime_exp"]
            lime_exp.as_pyplot_figure()
            plt.title("Explicabilidad global de LIME")
            
            # Ajustar el tamaño de las etiquetas del eje Y
            plt.yticks(fontsize=10)  # Tamaño de letra más pequeño para las etiquetas del eje Y

            # Ajustar el espaciado de los nombres de las características
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            plt.subplots_adjust(left=0.4)
            
            plt.xlabel('Valor de impacto', fontsize=10)
            plt.ylabel('Características', fontsize=10)
            
            plt.show()


    def calcular_impacto_por_interseccion(self):
        """
        Calcula el impacto total de cada característica en cada intersección.

        shap_values_dict: diccionario con estructura {intersección: {característica: valores SHAP}}
                        (los valores SHAP pueden ser un array con valores para cada clase de salida)

        Retorna un diccionario con el impacto promedio por intersección y característica.
        """
        impacto_intersecciones = {}
        print("El diccionario es: {self.dic_map}")

        for interseccion, caracteristicas in self.dic_map.items():
            impacto_intersecciones[interseccion] = {
                feature: np.mean(shap_values)  # Promedio absoluto del impacto de la característica
                for feature, shap_values in caracteristicas.items()
            }

        return impacto_intersecciones

    def guardar_explicabilidad_txt(self, directorio, model):
        """
        Guarda un archivo TXT para cada característica con el impacto en cada intersección.
        
        impacto_intersecciones: dict con {interseccion: {caracteristica: impacto}}
        directorio: ruta donde se guardarán los archivos
        carpeta: subcarpeta donde almacenar los txt
        """
        
        if model == "pytorch":       
            path_completo = os.path.join(directorio, "mapas_explicabilidad_txt_pytorch")
            os.makedirs(path_completo, exist_ok=True)  
        elif model == "sklearn":
            path_completo = os.path.join(directorio, "mapas_explicabilidad_txt_sklearn")
            os.makedirs(path_completo, exist_ok=True) 
        
        impacto_intersecciones = self.calcular_impacto_por_interseccion()
        

        # Transponer el diccionario para agrupar por característica en lugar de intersección
        caracteristicas = set()
        for impactos in impacto_intersecciones.values():
            caracteristicas.update(impactos.keys())
            

        # Escribir un archivo para cada característica
        for feature in caracteristicas:
            file_path = os.path.join(path_completo, f"{feature}.txt")
            with open(file_path, "w") as f:
                for interseccion, impactos in impacto_intersecciones.items():
                    if feature in impactos:
                        f.write(f"Intersección {interseccion}: {impactos[feature]:.6f}\n")

        print(f"Archivos guardados en: {path_completo}")




