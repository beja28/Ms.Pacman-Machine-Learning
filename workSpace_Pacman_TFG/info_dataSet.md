# DATASET

#### Se guardan los estados en los que Pacman pasa por una intersección

<br>

##### Columnas del estado del juego que he decidido quitar



- He decidido quitar la variable de "*currentLevelTime*" y "*levelCount*"
- He decidido quitar la variable de "*pillsState*" que devuelve una secuencia de 250 1s y 0s indicando si una pill concreta se ha comido
- He decidido quitar la variable de "*pillWasEaten*" y "*powerPillWasEaten*"
- He decidido quitar la variable de "*powerPillsState*" ya que devuelve una secuecia de 4 1s y 0s, y no sirve esa representacion
- He decidido quitar la variable de "*pacmanLivesRemaining*" no sirve de nada saber cuantas vidas quedan
- He decidido quitar la variable de "*pacmanReceivedExtraLife*" ya que solo se recive cuando se alcanzan los 10000 puntos y no afecta en nada
- He decidido quitar la variable de "*mazeIndex*" ya que es el indice del tablero que se esta jugando actualmente y de momento solo vamos a jugar con el primer nivel


<br>

##### Columnas del estado del juego que he decidido añadir


- He decidido añadidir 3 columnas adicionales ("*scoreDiffX*"), con la diferencia de las puntuaciones obtenidas en los 10, 25 y 50 anteriores ticks de ejecucion (-1 en el caso de que no se pueda calcular)
- He decidido añadidir 4 columnas ("*ghostXDistance*"), guardando la distancia del "path mas corto" a los fantasmas (-1 si esta en la carcel)
- He decidido añadidir 2 columnas ("*euclideanDistanceToPp*" y "*pathDistanceToPp*") indicando la distancia a la PP mas cercana "del path mas corto" y la distancia "euclidea " (-1 si no existe ninguna activa)
- He decidido añadidir 1 columna ("*remainingPp*"), con el numero de PP restantes. Puede ser util para saber si Pacman tiene que jugar mas agresivo o mas defensivo



<br>

##### Columnas que faltan de calcular
- Añadir 1 columna con la distancia del path mas corto a la pill mas cercana (puede ser util cuando ya no quedan PP y no hay fantasmas cerca)


<br>

##### Características que se guardan finalmente por cada estado


| Variable                   | Descripción                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| PacmanMove                 | Movimiento actual realizado por Pacman.                                      |
| totalTime                  | Tiempo total transcurrido en el juego.                                       |
| score                      | Puntuación actual del juego.                                                |
| pacmanCurrentNodeIndex      | Índice del nodo actual en el que se encuentra Pacman.                        |
| pacmanLastMoveMade          | Último movimiento realizado por Pacman.                                     |
| ghost1NodeIndex             | Índice del nodo actual del primer fantasma.                                 |
| ghost1EdibleTime            | Tiempo restante en el que el primer fantasma es comestible.                 |
| ghost1LairTime              | Tiempo que el primer fantasma debe pasar en la carcel  antes de salir.       |
| ghost1LastMove              | Último movimiento realizado por el primer fantasma.                         |
| ghost2NodeIndex             | Índice del nodo actual del segundo fantasma.                                |
| ghost2EdibleTime            | Tiempo restante en el que el segundo fantasma es comestible.                |
| ghost2LairTime              | Tiempo que el segundo fantasma debe pasar en la carcel  antes de salir.     |
| ghost2LastMove              | Último movimiento realizado por el segundo fantasma.                        |
| ghost3NodeIndex             | Índice del nodo actual del tercer fantasma.                                 |
| ghost3EdibleTime            | Tiempo restante en el que el tercer fantasma es comestible.                 |
| ghost3LairTime              | Tiempo que el tercer fantasma debe pasar en la carcel  antes de salir.      |
| ghost3LastMove              | Último movimiento realizado por el tercer fantasma.                         |
| ghost4NodeIndex             | Índice del nodo actual del cuarto fantasma.                                 |
| ghost4EdibleTime            | Tiempo restante en el que el cuarto fantasma es comestible.                 |
| ghost4LairTime              | Tiempo que el cuarto fantasma debe pasar en la carcel antes de salir.      |
| ghost4LastMove              | Último movimiento realizado por el cuarto fantasma.                         |
| timeOfLastGlobalReversal    | Tiempo en el que ocurrió la última inversión global.         |
| pacmanWasEaten              | Indica si Pacman fue comido (booleano).                                     |
| ghost1Eaten                 | Indica si el primer fantasma fue comido (booleano).                         |
| ghost2Eaten                 | Indica si el segundo fantasma fue comido (booleano).                        |
| ghost3Eaten                 | Indica si el tercer fantasma fue comido (booleano).                         |
| ghost4Eaten                 | Indica si el cuarto fantasma fue comido (booleano).                         |
| scoreDiff10                 | Diferencia en la puntuación de los últimos 10 ticks.                        |
| scoreDiff25                 | Diferencia en la puntuación de los últimos 25 ticks.                        |
| scoreDiff50                 | Diferencia en la puntuación de los últimos 50 ticks.                        |
| ghost1Distance              | Distancia al primer fantasma desde Pacman.                                  |
| ghost2Distance              | Distancia al segundo fantasma desde Pacman.                                 |
| ghost3Distance              | Distancia al tercer fantasma desde Pacman.                                  |
| ghost4Distance              | Distancia al cuarto fantasma desde Pacman.                                  |
| euclideanDistanceToPp       | Distancia "euclidea" de Pacman, a la Power Pill más cercana.                           |
| pathDistanceToPp            | Distancia del "path", de Pacman a la Power Pill más cercana.               |
| remainingPp                 | Número de Power Pills restantes en el juego.                                |

