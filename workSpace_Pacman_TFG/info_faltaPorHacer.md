## FALTA DE HACER (por prioridad)


##### CREAR una nueva clase "ExecutorConectionNN" que se encarga de conectarse através de un socket con python
- Las clases que se necesiten crear para el socket, se tienen crear en la **carpeta "socket"** del game
- Se pasa un estado del juego modificado, atraves del socket
- Se procesa y se calcula el resultado con una red Neuronal
- Se recibe através del socket el movimiento que debe realizar Pacman
- Pacman ejecuta ese movimiento



<br>

##### Cosas de menor importancia

- Repasar todo el codigo y cambiarlo un poco, xq hay funciones copiadas del chat
- Modificar la clase de ExecutorGenerateDataSet para que reciba el numero de ejecuciones que se quiere hacer
Conseguir que cada vez que se genera un DataSet se guarde informacion relaccionada
- Si es la primera vez que se crea el DataSet, se debe crear un archivo .txt y añadir la informacion 
- Si ya hay un DataSet creado y se modifica (se añaden nuevas filas) --> Se debe añadir al .txt relaccionado con ese dataSet
		  una nueva fila con la fecha de la modificacion y las nuevas filas que se han añadido, y el pacman y lso fantasmas usado
- Calcular variables nuevas




<br>

##### COSAS QUE REVISAR DATASET

- Revisar la columna de "*pacmanLastMoveMade*" ns si esta devolviendo lo correcto



<br>

##### DUDAS
- De momento no he hecho DataSets mezclando distintos fantasmas y Pacmans