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
   ```bash
   python -m pip install pytorch-tabnet==4.1
   ```
### Preparación del Workspace en Java (Eclipse)

Tanto para la preparación del workspace como para ejecutar el modelo final ha de estar en la rama `multi-network`.

#### 1. Crear el Workspace
Crea una **nueva carpeta vacía** que funcionará como tu workspace de Eclipse.

#### 2. Copiar los Proyectos
Desde el proyecto descargado, copia las siguientes carpetas al workspace creado:

- `Evaluation`
- `PacManEngine`
- `PacmanNeuro`

#### 3. Iniciar Eclipse
Abre **Eclipse** y, cuando se te solicite, selecciona el **nuevo workspace** que creaste.

#### 4. Importar los Proyectos al Workspace
Sigue estos pasos para importar los proyectos:

1. Ve a: `File` → `Import`.
2. Selecciona: `General` → `Existing Projects into Workspace`.
3. Haz clic en **Browse** y selecciona la carpeta del workspace.
4. Marca los proyectos detectados.
5. Haz clic en **Finish**.

#### 5. Importar Manualmente Proyectos No Detectados
Si alguna carpeta no se importa automáticamente, haz lo siguiente:

1. Ve a: `File` → `New` → `Java Project`.
2. Desmarca la opción: **Use default location**.
3. Haz clic en **Browse** y selecciona la carpeta del proyecto que falta.
4. Desactiva la opción: **Create module-info.java file**.
5. Haz clic en **Finish**.

---

> Asegúrate de que todos los proyectos estén correctamente configurados como proyectos Java dentro de Eclipse para evitar errores de compilación.


### Cómo ejecutar el modelo final

#### 1. Descargar el dataset empleado para cada experimento
- Consulta el [Dataset](DataSets/info_datasets_modelos.md) empleado para cada experimento.
- Descarga el Dataset correspondiente y guardalo en la carpeta Datasets.
   - https://drive.google.com/drive/folders/1JAVIr6Yl94kzg011YRiNgcJwPQtBpZ0N?usp=drive_link
  

#### 2. Seleccionar el modelo tabnet
```bash
cd python/codigo
```
```bash
python .\main.py model tabnet
```

#### 3. Ejecutar la partida desde Eclipse

- En Evaluation, seleccione la clase **ExecutorTest**: `Run As` → `Java Application`

#### 4. Posible error al ejecutar ExecutorTest
Al ejecutar la clase `ExecutorTest`, podría aparecer un error como el siguiente:

      Exception in thread "main" java.lang.ExceptionInInitializerError
      Caused by: java.lang.NullPointerException
      at pacman.game.internal.Maze.loadNodes(Maze.java:42)
      at pacman.game.Game.<clinit>(Game.java:77)

Para solucionarlo ve a `Project` → `Clean`, y seleccione todos los proyectos.

### Cómo probar los primeros modelos
Recuerda seleccionar el Dataset necesario para cada modelo.

#### Evaluación 1: Una sola red neuronal
1. Selecciona la rama `single-network` y prepara el workspace como se explica arriba.
2. Para ejecutar los modelos:
```bash
python .\main.py model <arquitecture>
```   
- <arquitecture> Puede ser pytorch o sklearn, dependiendo del modelo que quieras usar.
3. En `ExecutorTest`: `Run As` → `Java Application`

#### Evaluación 2: Una red por intersección con Scikit-Learn y Pytorch
1. Selecciona el tag `v2.0` y prepara el workspace como se explica arriba.
2. Para ejecutar los modelos:
```bash
python .\main.py model <arquitecture>
```   
- <arquitecture> Puede ser pytorch o sklearn, dependiendo del modelo que quieras usar.
3. En `ExecutorTest`: `Run As` → `Java Application`

#### Evaluación 3: Una red por intersección con Tabnet
1. Selecciona el tag `v3.0` y prepara el workspace como se explica arriba.
2. Para ejecutar los modelos:
```bash
python .\main.py model tabnet
```   
3. En `ExecutorTest`: `Run As` → `Java Application`

### Notas

- Asegúrate de tener una versión de Python compatible (3.8 o superior se recomienda).
- Si encuentras problemas al activar el entorno virtual en Windows, verifica que la ejecución de scripts está habilitada:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
  Esto permite ejecutar scripts locales.
