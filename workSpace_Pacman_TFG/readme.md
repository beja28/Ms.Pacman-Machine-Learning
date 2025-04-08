# Workspace Java


Este workspace contiene tres proyectos principales en Java, organizados de la siguiente manera:

### 1. **Evaluation**
   - Contiene los archivos `main` para ejecutar distintos modos de ejecución.
   - Permite evaluar distintas configuraciones y métodos de funcionamiento.
   - Incluye archivos de configuración y pruebas.
   
   **Estructura del proyecto:**
   ```
   Evaluation/
   ├── src/
   │   ├── (default package)/
   │   │   ├── Evaluate.java
   │   │   ├── ExecutorTest.java
   │   │   ├── ExecutorTestDataSet.java
   │   │   ├── ExecutorTestHeatMaps.java
   │   │   ├── ExecutorTestMultiDataSet.java
   │   │   ├── ExecutorTestSocketConnection.java
   │   ├── config.properties
   ├── mapas_explicabilidad_txt_pytorch/
   ├── mapas_explicabilidad_txt_sklearn/
   ├── mapas_pytorch/
   ├── mapas_sklearn/
   ├── new_dataSets/
   ```

   **Modos de ejecución**:
   Explicacion detallada para cada modo de ejecución (de momento no lo pongo porque no son los finales)

### 2. **PacmanEngine**
   - Contiene toda la lógica del juego Pac-Man.
   - Se ubican aquí las clases relacionadas con el manejo de controladores y utilidades.
   - Se deben realizar modificaciones en esta estructura de carpetas para cambios en la lógica del juego.
   - Se usa `Maven` para la gestión de dependencias y compilación.
   
   **Estructura del proyecto:**
   ```
   PacmanEngine/
   ├── src/main/java/
   │   ├── es.ucm.fdi.ici/
   │   │   ├── pacman/
   │   │   │   ├── controllers/
   │   │   │   ├── game/
   │   │   │   │   ├── comms/
   │   │   │   │   ├── consolePrinter/
   │   │   │   │   ├── dataManager/
   │   │   │   │   ├── heatmap/
   │   │   │   │   ├── info/
   │   │   │   │   ├── internal/
   │   │   │   │   ├── util/
   │   │   │   │   ├── Constants.java
   │   │   │   │   ├── Drawable.java
   │   │   │   │   ├── Game.java
   │   │   │   │   ├── GameObserver.java
   │   │   │   │   ├── GameView.java
   │   │   │   │   ├── Executor.java
   │   │   │   │   ├── ExecutorModes.java
   ├── src/main/resources/
   ├── src/test/java/
   ├── pom.xml
   ```

**Gestión de datos y creación de datasets**

Para realizar la recolección y almacenamiento de los datos de los estados del juego, se creó una carpeta dentro del directorio `game` llamada `dataManager`. En ella se encuentran tres clases encargadas de gestionar estos datos:

- **DataSetRecorder**: Recopila y almacena los estados válidos del juego, es decir, aquellos en los que Pac-Man se encuentra en una intersección del tablero. Procesa estos estados mediante la clase `GameStateFilter` y gestiona la escritura de datos en memoria en formato CSV.
- **GameStateFilter**: Procesa el estado del juego, calculando nuevas variables relevantes y eliminando aquellas que no lo son.
- **DataSetVariables**: Gestiona listas de `strings` con los nombres de las características del juego original, aquellas que deben eliminarse y las nuevas que se deben añadir.


### 3. **PacmanNeuro**
   - Implementa la funcionalidad de Pac-Man con redes neuronales.
   - Se conecta con Python a través de un socket para interactuar con modelos de redes neuronales entrenados en python previamente.
   - Contiene clases específicas para gestionar la comunicación con Python, enviar los estados filtrados del juego en un instante y recbir el movimiento a realizar calculado por la red neuronal.
   
   **Estructura del proyecto:**
   ```
   PacmanNeuro/
   ├── src/
   │   ├── es.ucm.fdi.ici.TFGpacman/
   │   │   ├── GhostsRandom.java
   │   │   ├── PacmanNeuro.java
   │   │   ├── SocketPython.java
   ```

## Requisitos
Para utilizar este workspace, asegúrate de contar con los siguientes requisitos:
- **Java JDK** (versión recomendada: 11 o superior)
- **Eclipse IDE for Java Developers** (o cualquier otro IDE compatible con Java)
- **Maven** (necesario para el proyecto `PacmanEngine`)

## Configuración y Uso
1. **Clonar el repositorio (si es necesario)**
   ```bash
   git clone -b main --single-branch <URL_DEL_REPOSITORIO>
   ```
2. **Abrir Eclipse y configurar el workspace**
   - Ejecutar Eclipse
   - Seleccionar la ubicación del workspace cuando lo solicite
3. **Importar proyectos**
   - Ir a `File > Import > Existing Projects into Workspace`
   - Seleccionar la carpeta del workspace
   - Importar los proyectos detectados
4. **Ejecutar un proyecto**
   - Seleccionar la clase con el método `main` en `Evaluation` para probar distintas ejecuciones.
   - Para probar la implementación con redes neuronales, asegúrate de ejecutar previamente un modelo de predicción en Python y ejecuta `PacmanNeuro` con la configuración adecuada.
