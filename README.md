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
   python -m pip install pytorch-tabnet==4.1
   ```
   ```bash
   pip install -r requirements.txt
   ```
### Preparación del workspace en Java
   En un entorno como Eclipse haga lo siguiente:
   #### 1. Cree una carpeta nueva que será usada como workspace.
   #### 2. Copie las 3 carpetas (Evaluation, PacManEngine, PacmanNeuro) del proyecto descargado y peguelas en el workspace creado.
   #### 3. Inicie Eclipse seleccionando el nuevo workspace.
   #### 4. Importe la carpeta del workspace:
      Import Proyects -> General -> Existing Proyects into Workspace -> Browse
   #### 5. Si no se ha importado alguna carpeta:
      1. File
      2. New
      3. Java Proyect
      4. Remove check on Use default location
      5. Browse
      6. Select folder missing
      7. Remove Create module-info.java file
      8. Finish

### Cómo ejecutar el modelo final

#### 1. **Seleccionar el modelo tabnet**

```bash
python main.py model tabnet
```

#### 2. **Ejecutar la partida desde Eclipse**

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
