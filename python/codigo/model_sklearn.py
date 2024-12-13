import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score
from datetime import datetime
import os

class MLPModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', batch_size=100, learning_rate_init=0.001, max_iter=500, random_state=333):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state
        )

    def train_and_cross_validate(self, X, Y, cv=5):

        print("Empiezo a entrenar MLP con validación cruzada.")
        scoring = {'accuracy': make_scorer(accuracy_score)}
        cv_results = cross_validate(self.mlp, X, Y, cv=cv, scoring=scoring, return_train_score=True)

        # Mostrar resultados de validación cruzada
        print(f"Train Accuracy (mean): {cv_results['train_accuracy'].mean():.4f}")
        print(f"Test Accuracy (mean): {cv_results['test_accuracy'].mean():.4f}")
        
        # Entrenar el modelo en el conjunto completo de datos para guardarlo
        self.mlp.fit(X, Y)
        print("Entrenamiento completo en todo el conjunto de datos.")

        return cv_results

    def save_model_mlp(self, path_trained, key):
        
        # Crear una carpeta con la fecha actual
        date_str = datetime.now().strftime('%Y-%m-%d')
        folder_path = os.path.join(path_trained, f'models_{date_str}')
        os.makedirs(folder_path, exist_ok=True)
        
        # Crear el nombre del archivo incluyendo el identificador 'key'
        model_filename = f'mlp_trained_model_{date_str}_{key}.pkl'
        final_path = os.path.join(folder_path, model_filename)
        
        # Guardar el modelo en un archivo
        joblib.dump(self.mlp, final_path)
        print(f'Modelo MLP guardado en {final_path}')

    
    def load_model_mlp(filepath):
        """
        Carga un modelo MLP desde un archivo .pkl.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"El archivo {filepath} no existe.")
        print(f"Cargando el modelo desde {filepath}...")
        return joblib.load(filepath)