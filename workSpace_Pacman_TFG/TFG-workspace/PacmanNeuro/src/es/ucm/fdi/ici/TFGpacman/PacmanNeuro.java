package es.ucm.fdi.ici.TFGpacman;


import java.util.Arrays;

import pacman.controllers.PacmanController;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import pacman.game.dataManager.TickRecorder;

public class PacmanNeuro extends PacmanController{

	
	private SocketPython socketPython;
	private TickRecorder recorder;
	
	public PacmanNeuro() {
		// Crear instancia de SocketPython
		try {
			socketPython = new SocketPython("localhost", 12345);
		} catch (Exception e) {
			System.out.println(e.getMessage());
			System.out.println("Error al inicializar el socket");
		}
		
		recorder = null;
	}
	
	
    @Override
    public MOVE getMove(Game game, long timeDue) {
    	
    	if(recorder == null) {
    		recorder = new TickRecorder(game);
    	}
    	
    	MOVE pacmanMove = MOVE.NEUTRAL;
    	
    	if (game.isJunction(game.getPacmanCurrentNodeIndex())) {
			String st = recorder.collectTick();
			String response = socketPython.sendGameState(st);
			
			//Obtenemos los posibles movimientos en esa interseccion
			MOVE[] possibleMoves = game.getPossibleMoves(game.getPacmanCurrentNodeIndex());

			try {
				MOVE predictedMove = MOVE.valueOf(response);				
				
				//Se comprueba que sea valido
			    boolean esMovimientoValido = Arrays.asList(possibleMoves).contains(predictedMove);
			    if (!esMovimientoValido) {
			    	System.out.println("[ERROR] Movimiento recibido por el modelo de Red Neuronal es invalido");
			    }else {
			    	pacmanMove = predictedMove;
			    }
			} catch (Exception e) {
				System.out.println("[ERROR] Respuesta inválida del servidor");				
			}
		}

    	System.out.println("[INFO] El movimiento a realizar es: " + pacmanMove.toString());	
    	
		return pacmanMove;
	}    
    

	
	public String getName() {
		return "Pacman Neuronal";
	}

}
