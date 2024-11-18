import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

class explicabilidad:
    def ejecutar_explicabilidad(self, model, technique, X, Y=None):
        """Ejecuta la técnica de explicabilidad seleccionada.
        Args:
            model: El modelo entrenado (PyTorch o Scikit-Learn).
            technique (str): La técnica a aplicar ("shap", "feature_importance", "lime").
            data (DataFrame o np.array): Los datos a usar para la explicabilidad.
        """
        if technique == "shap":
            self.explicabilidad_shap(model, X)
        elif technique == "feature_importance":
            self.explicabilidad_feature_importance(model, X, Y)
        elif technique == "lime":
            self.explicabilidad_lime(model, X)
    
    # Calcula la contribución de cada característica en las predicciones del modelo
    def explicabilidad_shap(self, model, X):
        try:
            """Implementación de SHAP para explicar las predicciones del modelo."""
            if isinstance(model, torch.nn.Module): # Comprueba que tipo de modelo es
                background_data = X.sample(n=200)
                data_tensor = torch.tensor(background_data.values, dtype=torch.float32)
                
                # Configurar dispositivo
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                data_tensor = data_tensor.to(device)
                
                explainer = shap.DeepExplainer(model, data_tensor)
                
                data_to_explain = X.sample(n=10)
                data_to_explain_tensor = torch.tensor(data_to_explain.values, dtype=torch.float32).to(device)
                
                shap_values = explainer.shap_values(data_to_explain_tensor)
                shap.summary_plot(shap_values, data_to_explain, plot_type="bar")
            else:
                background_data = shap.sample(X, 200)
                explainer = shap.KernelExplainer(model.predict, background_data)

                data_to_explain = X.sample(n=100)
                shap_values = explainer.shap_values(data_to_explain) # Calcula los valores SHAP
                shap.summary_plot(shap_values, data_to_explain, plot_type="bar") # Gráfico de los datos SHAP

        except Exception as e:
            print(f"Error durante la explicabilidad SHAP: {e}")

    # Calcula la importancia de las características mediante permutaciones. (Solo Scikit-Learn)
    def explicabilidad_feature_importance(self, model, X, Y):
        """Implementación de Feature Importance utilizando Permutation Importance."""
        if hasattr(model, 'predict'): # Hay que comprobar que tiene método predict porque es necesario más adelante
            # Función que evalúa el impacto de cada característica
            X_sampled = X.sample(n=500, random_state=42)  # Muestra de 500 instancias
            Y_sampled = Y[X_sampled.index]  # Etiquetas correspondientes a las muestras
            print(f"Usando {X_sampled.shape[0]} muestras para Permutation Importance.")
            # n_repeats (cuántas veces se permutará cada característica. + grande / + resultado / + tmp computo)
            # random_state (semilla para poder repodrucir resultados)
            results = permutation_importance(
                model, 
                X_sampled, 
                Y_sampled, 
                n_repeats=5,  # Reducir repeticiones para mejorar rendimiento
                random_state=1
            )    # Calcula la permutación
            importances = results.importances_mean # Almaceno array de valores. + valor / + influencia

            feature_names = X.columns
            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            # Ordenar características por importancia
            feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
            print("Importancia de las características (ordenada):\n", feature_importances)

            # Visualizar resultados
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
            plt.gca().invert_yaxis()  # Invertir el eje para mostrar la más importante arriba
            plt.xlabel("Mean Importance")
            plt.ylabel("Features")
            plt.title("Feature Importance (Permutation)")
            plt.show()
        else:
            print("Feature importance solo se implementa para modelos de Scikit-Learn.")
    
    # Explica las predicciones del modelo en una muestra específica
    def explicabilidad_lime(self, model, X):
        """Implementación de LIME para explicar las predicciones del modelo."""
        # Asegurarse de que X tenga nombres de características
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=model.feature_names_in_)
        
        explainer = LimeTabularExplainer(
            X.values,  # Convertimos a array
            mode="classification",
            feature_names=X.columns,  # Nombres de columnas
            discretize_continuous=True
        )
        # Seleccionar una instancia aleatoria
        i = np.random.randint(0, len(X))
        instance = X.iloc[i]  # Selecciona una fila del DataFrame
        
        print(f"Explicando la instancia {i}")
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            raise NotImplementedError("El modelo debe implementar predict_proba para LIME.")
        
        # Generar explicación
        exp = explainer.explain_instance(instance.values, predict_fn, num_features=5)

        # Mostrar resultados en consola
        print("Explicación:")
        for feature, importance in exp.as_list():
            print(f"{feature}: {importance}")
        
        # Guardar visualización
        exp.as_pyplot_figure()
        plt.show()



