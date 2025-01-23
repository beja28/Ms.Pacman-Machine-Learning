# Ms.Pacman-Machine-Learning

## Requisitos del Proyecto

Este repositorio incluye un archivo `requirements.txt` que especifica las versiones necesarias de las librerías utilizadas. Esto permite crear un entorno virtual consistente para trabajar con el proyecto.

### Cómo configurar el entorno

1. **Crear un entorno virtual**:
   Abre una terminal en la raíz del proyecto y ejecuta:

   ```bash
   python -m venv .venv
   ```

   También puedes abrir la paleta de comandos en VSCode (Ctrl + Shift + P)
   y buscar "Python: Create Terminal" para abrir una terminal con el entorno virtual activado.

   Seleccionar la versión de Python correspondiente al entorno virtual creado, y luego seleccionar el archivo requirements.txt para instalar las dependencias.

2. **Activar el entorno virtual**:

   - En Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Instalar las dependencias**:
   Con el entorno virtual activado, instala las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```

### Cómo ejecutar el programa

El programa incluye diferentes comandos que permiten entrenar modelos, realizar predicciones y aplicar técnicas de explicabilidad. A continuación, se detallan los comandos disponibles:

#### 1. **Seleccionar un modelo para predicción**

```bash
python main.py model <model>
```

- `<model>`: Puede ser `pytorch` o `sklearn`, dependiendo del modelo que quieras usar.

**Ejemplo**:

```bash
python main.py model pytorch
```

#### 2. **Entrenar un modelo**

```bash
python main.py train_model <train_model>
```

- `<train_model>`: Puede ser `pytorch` o `sklearn`, según el modelo que desees entrenar.

**Ejemplo**:

- Entrenar el modelo PyTorch:
  ```bash
  python main.py train_model pytorch
  ```
- Entrenar el modelo Scikit-Learn:
  ```bash
  python main.py train_model sklearn
  ```

#### 3. **Realizar explicabilidad**

```bash
python main.py explain <model> <technique>
```

- `<model>`: Selecciona el modelo a explicar (`pytorch` o `sklearn`).
- `<technique>`: Técnica de explicabilidad a aplicar (`shap`, `feature_importance`, o `lime`).

**Ejemplos**:

- Explicar el modelo PyTorch usando SHAP:
  ```bash
  python main.py explain pytorch shap
  ```
- Explicar el modelo Scikit-Learn usando LIME:
  ```bash
  python main.py explain sklearn lime
  ```

### Notas

- Asegúrate de tener una versión de Python compatible (3.8 o superior se recomienda).
- Si encuentras problemas al activar el entorno virtual en Windows, verifica que la ejecución de scripts está habilitada:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
  Esto permite ejecutar scripts locales.
