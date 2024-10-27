## CAMBIOS QUE HE REALIZADO EN EL WORKSPACE

- He eliminado todos los proyectos de ICI que no nos interesan
- He cambiado los nombres y refactorizado el proyecto
- He creado una **nueva carpeta dentro del "*game*" llamada "*dataset*"** que contiene las nuevas clases que he creado para recopilar
	  los distintos estados del juego, y crear el dataset en formato ".csv"
- He creado una **nueva carpeta dentro del "*game*" llamada "*socket*"** que contendrá lo necesario para tener el socket
- He creado una **nueva clase "*ExecutorGenerateDataSet*"** similar a la clase existente de "*Executor*" que se encarga de guardar los estados **filtrados** de las distintas ejecuciones del juego en un archivo .csv
- He creado una **nueva clase "*ExecutorScoketConection*"** similar a la clase existente de "*Executor*" que se encarga de **establecer la conexión através del socket con python**. Además envía la informacion de los estados del juego através del socket y **recibe el movimiento que debe realizar Pacman**
