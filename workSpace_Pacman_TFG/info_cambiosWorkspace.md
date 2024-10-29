## CAMBIOS QUE HE REALIZADO EN EL WORKSPACE

- He eliminado todos los proyectos de ICI que no nos interesan
- He cambiado los nombres y refactorizado el proyecto
- He creado una **nueva carpeta dentro del "*game*" llamada "*dataManager*"** que contiene las nuevas clases que he creado para recopilar los distintos estados del juego, y crear el dataset en formato ".csv"
	- Clase ***DataSetRecorder*** --> se encarga de guardar los estados válidos del juego, procesarlos y guardarlos en un archivo .csv
	- Clase ***DataSetVariables*** --> contiene los nombres de las variables que queremos eliminar y calcular. Además de funciones que modifican listas de Strings
	- Clase ***GameStateFilter*** --> se encarga de procesar un estado del juego: elimina datos que no queremos y calcula nuevos datos 
- He creado una **nueva carpeta dentro del "*game*" llamada "*socket*"** que contiene una clase llamada ***"socketPython"***, que se encarga de establecer la conexión con un puerto del sistema. Cuenta con funciones para **mandar y recibir información** através del socket
- He creado una **nueva clase "*ExecutorModes*"** similar a la clase existente de "*Executor*" que contiene las funciones necesarias para ejecutar:
	- Modo de **creación del DataSet *"runGameGenerateDataSet":*** se encarga de guardar los estados **filtrados** de las distintas ejecuciones del juego en un archivo .csv
	Además, recibe el *numero de ejecuciones* que se desea realizar, el *nombre del arhivo* donde se quieren guardar los datos, y si se quiere *mostrar información durante la ejecución*
	- Modo de **conexión con Python a través del socket *"runGameSocketConection:"*** se encarga de **establecer la conexión através del socket con python**. Además envía la información de los estados del juego através del socket y **recibe el movimiento que debe realizar Pacman**
