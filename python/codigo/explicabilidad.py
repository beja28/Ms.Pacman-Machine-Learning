import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import os


class Explicabilidad:
    def __init__(self):
        self.explicaciones = []  # Almacena resultados de explicabilidad

    def ejecutar_explicabilidad(self, model, model_filename, technique, X, Y=None):
        """Ejecuta la técnica de explicabilidad seleccionada."""
        if technique == "shap":
            self.explicabilidad_shap(model, model_filename, X)
        elif technique == "feature_importance":
            self.explicabilidad_feature_importance(model, model_filename, X, Y)
        elif technique == "lime":
            self.explicabilidad_lime(model, model_filename, X)
    
    def explicabilidad_shap(self, model, model_filename, X):
        """Implementación de SHAP para explicar las predicciones del modelo."""
        try:
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
    
    def explicabilidad_lime(self, model, model_filename, X):
        """Implementación de LIME para explicar las predicciones del modelo (funciona para Scikit-Learn y PyTorch)."""
        try:
            # Si X no es un DataFrame, conviértelo en uno para manejar las características adecuadamente
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=model.feature_names_in_)

            # Crea el explicador de LIME
            explainer = LimeTabularExplainer(
                X.values,
                mode="classification",
                feature_names=X.columns,
                discretize_continuous=True
            )
            
            # Selecciona una fila aleatoria de X para explicarla
            i = np.random.randint(0, len(X))
            instance = X.iloc[i]  # Obtén una fila (instancia) del DataFrame
            
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
                "feature_names": X.columns
            })

        except Exception as e:
            print(f"Error durante la explicabilidad LIME: {e}")
        
        
    def generar_grafico_explicabilidad_global(self, directorio_actual, model):
        """Genera un gráfico que combine la explicabilidad global de todas las redes y lo guarda en un archivo."""

        images_dir = os.path.join(directorio_actual, '..', 'images')
        images_dir = os.path.normpath(images_dir)
        
        # Crear la ruta completa para la carpeta 'Explicabilidad' dentro de 'images'
        explicabilidad_dir = os.path.join(images_dir, 'Explicabilidad')
        os.makedirs(explicabilidad_dir, exist_ok=True)

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
          
            # Reducir `mean_shap_values` a una dimension
            mean_shap_values_reduced = np.mean(mean_shap_values, axis=1)
                       
            # Ordenar por la importancia (de mayor a menor)
            sorted_idx = np.argsort(mean_shap_values_reduced)[::-1]

            # Ordenar nombres y valores
            sorted_feature_names = feature_names[sorted_idx]
            sorted_mean_shap_values = mean_shap_values_reduced[sorted_idx]

            # Crear gráfico de barras
            plt.figure(figsize=(10, 6))
            plt.barh(sorted_feature_names, sorted_mean_shap_values, color='skyblue')
            plt.gca().invert_yaxis()  # Invertir el eje Y para que el de mayor importancia quede arriba
            plt.xlabel("Importancia media de SHAP")
            plt.ylabel("Características")
            plt.title("Importancia global de características (SHAP)")
            
            # Ajustar el tamaño de las etiquetas del eje Y
            plt.yticks(fontsize=8)  # Tamaño de letra más pequeño para las etiquetas del eje Y

            # Ajustar el espaciado de los nombres de las características
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

            # Guardar gráfico
            
            if model == "pytorch":
                shap_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_shap_pytorch.png")
            elif model == "sklearn":
                shap_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_shap_sklearn.png")
                     
            plt.savefig(shap_plot_path)  # Guardar gráfico como imagen
            plt.close()  # Cerrar la figura para evitar que se muestre

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
                plt.figure(figsize=(10, 6))
                importancias_mean.plot(kind='barh', color='skyblue')
                plt.gca().invert_yaxis()
                plt.xlabel("Mean Importance")
                plt.ylabel("Features", fontsize=5)
                plt.title("Importancia Global de Características")

                # Ajustar el tamaño de las etiquetas del eje Y
                plt.yticks(fontsize=8)  # Tamaño de letra más pequeño para las etiquetas del eje Y

                # Ajustar el espaciado de los nombres de las características
                plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

                # Solo para sklearn
                importance_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_importancia_feature_sklearn.png")

                plt.savefig(importance_plot_path)  # Guardar gráfico como imagen
                plt.close()

        # Generar gráfico LIME global
        lime_exps = [exp for exp in self.explicaciones if exp["technique"] == "lime"]
        if lime_exps:
            # Usar el primer experimento de LIME, aunque podrías combinar los resultados si lo deseas
            lime_exp = lime_exps[0]["lime_exp"]
            lime_exp.as_pyplot_figure()
            plt.title("Explicabilidad global de LIME")
            
            # Ajustar el tamaño de las etiquetas del eje Y
            plt.yticks(fontsize=8)  # Tamaño de letra más pequeño para las etiquetas del eje Y

            # Ajustar el espaciado de los nombres de las características
            plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            plt.subplots_adjust(left=0.4)
            
            if model == "pytorch":
                lime_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_lime_pytorch.png")
            elif model == "sklearn":            
                lime_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_lime_sklearn.png")
                
            plt.savefig(lime_plot_path)  # Guardar gráfico como imagen
            plt.close()

        print("Las gráficas de explicabilidad se han guardado en el directorio 'images/Explicabilidad'.")


