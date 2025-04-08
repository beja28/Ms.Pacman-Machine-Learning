import pacman.ExecutorModes;
import pacman.controllers.GhostController;
import pacman.controllers.HumanController;
import pacman.controllers.KeyBoardInput;
import pacman.controllers.PacmanController;

import java.util.ArrayList;
import java.util.List;

import es.ucm.fdi.ici.TFGpacman.PacManNeuro;


/*	ICI HALL-OF-FAME

--- Best MsPacMan (score) --- 
es.ucm.fdi.ici.c2223.practica1.grupo06.MsPacMan (9958)
es.ucm.fdi.ici.c2122.practica1.grupo10.MsPacMan (7164)
es.ucm.fdi.ici.c2324.practica1.grupo07.MsPacMan (5378)


es.ucm.fdi.ici.c2223.practica2.grupo02.MsPacMan (7000) 
es.ucm.fdi.ici.c2122.practica2.grupo01.MsPacMan (6517) 
es.ucm.fdi.ici.c2324.practica2.grupo01.MsPacMan (4452) 


--- Best Ghosts (score) --- 
es.ucm.fdi.ici.c2223.practica1.grupo06.Ghosts (2064)
es.ucm.fdi.ici.c2122.practica1.grupo01.Ghosts (2108)
es.ucm.fdi.ici.c2324.practica1.grupo08.Ghosts (2152)

es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts (1924)
es.ucm.fdi.ici.c2122.practica2.grupo01.Ghosts (2648)
es.ucm.fdi.ici.c2223.practica2.grupo02.Ghosts (3704)


 */



public class ExecutorTestAverageScore {

    public static void main(String[] args) {
    	ExecutorModes executor = new ExecutorModes.Builder()
                .setTickLimit(4000)
                .setVisual(true)
                .setScaleFactor(2.5)
                .build();

        PacmanController pacMan = new PacManNeuro();
        List<GhostController> ghostControllers = new ArrayList<>();
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica1.grupo06.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica1.grupo01.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica1.grupo08.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica2.grupo01.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica2.grupo02.Ghosts());
        
        
        /**
         * Ejecuta una serie de partidas entre un controlador de Pac-Man y una lista de controladores de fantasmas,
         * y calcula estadísticas sobre las puntuaciones obtenidas.
         * 
         * @param pacManController  Controlador de Pac-Man
         * @param ghostControllers  Lista de Controladores de los fantasmas
         * @param iter              Número de partidas a jugar por cada combinación de controladores
         * @param delay             Tiempo de espera (ms) entre cada partida
         * @param fileName          Nombre del archivo .txt donde se guardan las estadísticas (si es vacío "", no se guarda)
         * 
         * La función realiza las siguientes acciones:
         * - Almacena la puntuación final de cada partida
         * - Calcula estadísticas descriptivas (media, mediana, desviación típica, percentiles, skewness, curtosis, etc.)
         * - Muestra las estadísticas por consola
         * - Guarda las estadísticas en un archivo .txt en la carpeta 'statistics', si se ha especificado un nombre de archivo
         */
        executor.runGameCalculateAverageScore(pacMan, ghostControllers, 100, 1000, "estadisticas_sklearn");
    }
}
