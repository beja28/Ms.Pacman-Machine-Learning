import java.util.ArrayList;
import java.util.List;

import pacman.ExecutorModes;
import pacman.controllers.GhostController;
import pacman.controllers.HumanController;
import pacman.controllers.KeyBoardInput;
import pacman.controllers.PacmanController;

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



public class ExecutorTestMultiDataSet {

    public static void main(String[] args) {
    	ExecutorModes executor = new ExecutorModes.Builder()
                .setTickLimit(4000)
                .setVisual(false)
                .setScaleFactor(2.5)
                .build();

    	
    	List<PacmanController> pacManControllers = new ArrayList<>();
        pacManControllers.add(new es.ucm.fdi.ici.c2223.practica1.grupo06.MsPacMan());
        pacManControllers.add(new es.ucm.fdi.ici.c2223.practica1.grupo10.MsPacMan());
        pacManControllers.add(new es.ucm.fdi.ici.c2223.practica1.grupo07.MsPacMan());
        pacManControllers.add(new es.ucm.fdi.ici.c2223.practica2.grupo02.MsPacMan());
        List<GhostController> ghostControllers = new ArrayList<>();
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica1.grupo06.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica1.grupo01.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica2.grupo04.Ghosts());
        ghostControllers.add(new es.ucm.fdi.ici.c2324.practica2.grupo01.Ghosts());
        
        
        int ITERS = 100;
        String fileName = "pruebas_multi01";
        boolean DEBUG = false;
        int min_score = -1;
        
        
        
        
        /**
         * Ejecuta una serie de simulaciones del juego y genera un dataset en un archivo .csv
         * 
         * @param pacManController  Lista de Controladores de Pacman
         * @param ghostController   Lista de Controladores de los fantasmas
         * @param iter              Num de iteraciones o partidas a jugar por cada combinación de controladores
         * @param fileName          Nombre del archivo .csv donde se guardan los datos (si no existe se crea)
         * @param DEBUG             Indica si se quiere activar o desactivar el modo de depuracion
         * @param min_score         Puntos minimos requeridos para guardar los datos de una partida (-1 para desactivar el filtro)
         * 
         * La funcion realiza las siguientes acciones:
         * - Configura el entorno del juego
         * - Ejecuta iteraciones del juego con todas las combinaciones de los controladores proporcionados
         * - Almacena los estados del juego en un archivo .csv si se cumplen las condiciones
         * - Muestra informacion sobre el proceso de ejecucion, incluyendo el tiempo total de ejecucion 
         *   y el numero de líneas creadas en el dataset
         * - Si se activa el modo de DEBUG, imprime informacion detallada sobre cada iteracion
         */

        executor.runGameGenerateMultiDataSet(pacManControllers, ghostControllers, ITERS, fileName, DEBUG, min_score);     
    }
	
}
