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
        self.explicaciones_shap = []
        self.explicaciones_lime = []
        self.importancias_feature = []

    def ejecutar_explicabilidad(self, model, model_filename, technique, X, Y=None):
        """Ejecuta la técnica de explicabilidad seleccionada."""
        if technique == "shap":
            self.explicabilidad_shap(model, model_filename, X)
        elif technique == "feature_importance":
            self.explicabilidad_feature_importance(model, model_filename, X, Y)
        elif technique == "lime":
            self.explicabilidad_lime(model, model_filename, X)
    
    # Calcula la contribución de cada característica en las predicciones del modelo
    def explicabilidad_shap(self, model, model_filename, X):
        try:
            """Implementación de SHAP para explicar las predicciones del modelo."""
            shap_values_global = None
            if isinstance(model, torch.nn.Module): # Comprueba que tipo de modelo es
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
                shap_values = explainer.shap_values(data_to_explain) # Calcula los valores SHAP
                shap_values_global = shap_values 

            # Guardar resultados sin mostrar la gráfica
            self.explicaciones_shap.append(shap_values_global)
        
        except Exception as e:
            print(f"Error durante la explicabilidad SHAP: {e}")

    # Calcula la importancia de las características mediante permutaciones. (Solo Scikit-Learn)
    def explicabilidad_feature_importance(self, model, model_filename, X, Y):
        """Implementación de Feature Importance utilizando Permutation Importance."""
        if hasattr(model, 'predict'): # Hay que comprobar que tiene método predict porque es necesario más adelante
            X_sampled = X.sample(n=500, random_state=42)  # Muestra de 500 instancias
            Y_sampled = Y[X_sampled.index]  # Etiquetas correspondientes a las muestras
            
            print(f"Ejecuta la explicabilidad para el modelo {model_filename}")

            print(f"Usando {X_sampled.shape[0]} muestras para Permutation Importance.")
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
            self.importancias_feature.append(feature_importances)
        
        else:
            print("Feature importance solo se implementa para modelos de Scikit-Learn.")
    
    # Explica las predicciones del modelo en una muestra específica
    def explicabilidad_lime(self, model, model_filename, X):
        """Implementación de LIME para explicar las predicciones del modelo."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=model.feature_names_in_)

        explainer = LimeTabularExplainer(
            X.values,
            mode="classification",
            feature_names=X.columns,
            discretize_continuous=True
        )
        
        i = np.random.randint(0, len(X))
        instance = X.iloc[i]  # Selecciona una fila del DataFrame
        
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            raise NotImplementedError("El modelo debe implementar predict_proba para LIME.")
        
        print(f"Ejecuta la explicabilidad para el modelo {model_filename}")
        
        exp = explainer.explain_instance(instance.values, predict_fn, num_features=5)
        
        # Guardar resultados sin mostrar la gráfica
        self.explicaciones_lime.append(exp)
        
        
    def generar_grafico_explicabilidad_global(self, path_trained):
        """Genera un gráfico que combine la explicabilidad global de todas las redes y lo guarda en un archivo."""

        images_dir = os.path.join(path_trained, '..', 'images')
        images_dir = os.path.normpath(images_dir)
        
        # Crear la ruta completa para la carpeta 'Explicabilidad' dentro de 'images'
        explicabilidad_dir = os.path.join(images_dir, 'Explicabilidad')
        os.makedirs(explicabilidad_dir, exist_ok=True)

        # Generar gráfico SHAP global
        if self.explicaciones_shap:
            shap_values_all = np.concatenate(self.explicaciones_shap, axis=0)  # Concatenar todos los valores SHAP
            shap.summary_plot(shap_values_all, show=False)
            plt.title("Explicabilidad global de SHAP")
            shap_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_shap_global.png")
            plt.savefig(shap_plot_path)  # Guardar gráfico como imagen
            plt.close()  # Cerrar la figura para evitar que se muestre

        # Generar gráfico de Importancia de características global
        if self.importancias_feature:
            importancias = pd.concat(self.importancias_feature, axis=0)
            importancias_mean = importancias.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            importancias_mean.plot(kind='barh', color='skyblue')
            plt.gca().invert_yaxis()
            plt.xlabel("Mean Importance")
            plt.ylabel("Features")
            plt.title("Explicabilidad global de Feature Importance")
            importance_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_importancia_feature_global.png")
            plt.savefig(importance_plot_path)  # Guardar gráfico como imagen
            plt.close()

        # Generar gráfico LIME global
        if self.explicaciones_lime:
            lime_exp = self.explicaciones_lime[0]  # Se utiliza el primero, aunque podrías combinar todos si lo deseas
            lime_exp.as_pyplot_figure()
            plt.title("Explicabilidad global de LIME")
            lime_plot_path = os.path.join(explicabilidad_dir, "explicabilidad_lime_global.png")
            plt.savefig(lime_plot_path)  # Guardar gráfico como imagen
            plt.close()

        print("Las gráficas de explicabilidad se han guardado en el directorio 'images/Explicabilidad'.")
