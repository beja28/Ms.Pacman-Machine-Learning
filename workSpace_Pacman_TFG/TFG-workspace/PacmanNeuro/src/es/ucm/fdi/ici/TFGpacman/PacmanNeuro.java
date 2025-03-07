package es.ucm.fdi.ici.TFGpacman;


import java.util.Arrays;
import java.util.List;

import pacman.controllers.PacmanController;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import pacman.game.consolePrinter.MessagePrinter;
import pacman.game.dataManager.GameStateFilter;

public class PacManNeuro extends PacmanController{

	
	private SocketPython socketPython;
	private GameStateFilter gameStateFilter;
	private MessagePrinter printer;
	
	public PacManNeuro() {
		
		this.gameStateFilter = new GameStateFilter();
		this.printer = new MessagePrinter();
		
		try {
			socketPython = new SocketPython("localhost", 12345);
		} catch (Exception e) {
			System.out.println(e.getMessage());
			printer.mostrarError("Al inicializar el socket");
		}
	}
	
	
    @Override
    public MOVE getMove(Game game, long timeDue) {    	
    	
    	MOVE pacmanMove = MOVE.NEUTRAL;
    	
    	if (game.isJunction(game.getPacmanCurrentNodeIndex())) {
    		//Obtener estado del juego procesado
    		List<String> filteredState = gameStateFilter.filterGameState(game.getGameState());
    		List<String> finalState = gameStateFilter.addNewVariablesToFilteredState(game, filteredState);
					
    		//Respuesta de la red neuronal
			String response = socketPython.sendGameState(String.join(",", finalState));
			
			MOVE[] possibleMoves = game.getPossibleMoves(game.getPacmanCurrentNodeIndex());

			try {
				System.out.println(response);
				MOVE predictedMove = MOVE.valueOf(response);				
				
				//Se comprueba que sea valido
			    boolean esMovimientoValido = Arrays.asList(possibleMoves).contains(predictedMove);
			    if (!esMovimientoValido) {
			    	printer.mostrarError("Movimiento recibido por el modelo de Red Neuronal es invalido");
			    }else {
			    	pacmanMove = predictedMove;
			    }
			} catch (Exception e) {
				printer.mostrarError("Respuesta inválida del servidor");
			}
			
			printer.mostrarInfo("El movimiento a realizar es: " + pacmanMove.toString());
		}

		return pacmanMove;
	}    
    
	
	public String getName() {
		return "Paquita";
	}

}
