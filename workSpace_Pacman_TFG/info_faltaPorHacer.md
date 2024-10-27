## FALTA DE HACER (por prioridad)


##### SOCKET
- Completar el codigo de la clase **"socketConection"** que inicializa el socket, y tendrá una funcion para mandar mensajes por el socket, y para recibir
- El manejo de **excepciones** con try-cath tmb habrá que hacerlo dentro de esa clase
- Buscar una forma de separar las clases que filtran el estado del juego
 	- En clases que se usan para crear el DataSet
	- En clases que se usan para filtrar un estado del juego



<br>

##### Cosas de menor importancia

- Modificar la clase de *"ExecutorGenerateDataSet"* para que reciba el numero de ejecuciones que se quiere hacer y el nombre del archivo .csv
- Quitar las funcion de runGame que no se usan en la clase de *"ExecutorGenerateDataSet"* y "*ExecutorScoketConection*"
- Conseguir que cada vez que se genera un DataSet se guarde informacion relaccionada
	- Si es la primera vez que se crea el DataSet, se debe crear un archivo .txt y añadir la informacion 
	- Si ya hay un DataSet creado y se modifica (se añaden nuevas filas) --> Se debe añadir al .txt relaccionado con ese dataSet una nueva fila con la fecha de la modificacion y las nuevas filas que se han añadido, y el pacman y lso fantasmas usado
- Calcular variables nuevas




<br>

##### COSAS QUE REVISAR DATASET

- Revisar la columna de "*pacmanLastMoveMade*" ns si esta devolviendo lo correcto



<br>

##### DUDAS
- De momento no he hecho DataSets mezclando distintos fantasmas y Pacmans