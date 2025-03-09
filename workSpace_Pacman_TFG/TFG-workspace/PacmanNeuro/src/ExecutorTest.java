import es.ucm.fdi.ici.TFGpacman.GhostsRandom;
import es.ucm.fdi.ici.TFGpacman.PacManNeuro;
import pacman.Executor;
import pacman.controllers.GhostController;
import pacman.controllers.PacmanController;

public class ExecutorTest {

    public static void main(String[] args) {
    	
    	Executor executor = new Executor.Builder()
                .setTickLimit(4000)
                .setVisual(true)
                .setScaleFactor(2.5)
                .build();

    	
    	PacmanController pacMan = new PacManNeuro();
        GhostController ghosts = new GhostsRandom();
        
        executor.runGame(pacMan, ghosts, 50);
        
    }
	
}
