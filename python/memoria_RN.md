Move## MEMORIA


He leido el CSV con pandas

Tengo que transformar las variables categoricas en numericas, para podemos usar OneHotEncoder (lo usamos en IA2) o codificarlo de manera que cada movimiento tenga un numero asignado. (preguntar)

pacmanWasEaten, ghost1Eaten, ghost2Eaten, ghost3Eaten, ghost4Eaten: Estas son variables booleanas que podrían ser codificadas como 0 y 1, lo que no requiere codificación especial.

Tenemos que comprobar que es mejor para el OneHotEncoder si sparse = True o sparse = False.

- Si el CSV tiene muchas categorías únicas en una o más columnas, la codificación one-hot resultará en muchas columnas (por ejemplo, si una columna tiene 100 categorías, generarás 100 columnas),
	con True puede ser más eficiente en términos de memoria, ya que solo almacenará los valores no cero.
- Con sparse=True, el modelo almacenará los datos en una estructura de matriz dispersa, lo que ahorra memoria cuando hay muchas categorías, ya que solo se almacenan los índices y valores no cero.
- Si usas sparse=False, estarás creando una matriz densa en la que se almacenan todos los ceros explícitamente. Esto puede ser más pesado en memoria si tienes un gran número de categorías.
- Una matriz densa (con sparse=False) es más fácil de trabajar en muchas bibliotecas de machine learning y puede ser más conveniente para ciertas operaciones
- Las matrices dispersas pueden ser más lentas para algunas operaciones debido a la complejidad de manejar índices, pero en general, el rendimiento puede variar según la tarea específica que estés realizando.

A la hora de preprocesar los datos, las variables booleanas las he procesado poniendo a 0 cuando es False y 1 cuando es True

Despues convierto el dataset ya codificado en un tensor.

Tengo que separar mis variables en Caracteristicas y Etiquetas. Como las etiquetas son lo que va a predecir y aprender el modelo, voy a poner la variable nextMove, ya que es lo que me interesa saber al final,
	el resto de variables las usare para hacer las predicciones.

Preguntar si es interesante ir añadioendo todos los estudios de que funcion de perdida, activacion, optimizacion es mejor, e ir comparando resultados con graficas.
A la hora de entrenar en el bucle for, conviene separar en batches el contenido?

<br>

---

<br>

Hemos creado una nueva red neuronal a traves de mlp y ha sido entrenada
Ajustado el modelo simple de pytorch con batches y siguiendo la estructura de la practica de AA, tb tuve que mapear la variable nextMove para pasar los datos a un valor numerico.
Despues de eso entrene este modelo.

Tengo que mirar las diferencias que hay entre los datos de entrenamiento y los datos de cross validation (cv)

He creado dos funciones para guardar los resultados de entrenamiento con la fecha de este

Organización del codigo en modulos (main, models, preprocessing)
