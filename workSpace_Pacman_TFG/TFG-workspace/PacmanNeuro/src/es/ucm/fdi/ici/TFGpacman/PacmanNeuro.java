package es.ucm.fdi.ici.TFGpacman;


import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

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
					
            // Obtener movimientos posibles sin colisiones
            MOVE[] possibleMoves = game.getPossibleMoves(game.getPacmanCurrentNodeIndex());
            List<MOVE> validMoves = new ArrayList<>(Arrays.asList(possibleMoves));

            // Restringir la vuelta atrás
            MOVE lastMove = game.getPacmanLastMoveMade();
            MOVE oppositeMove = lastMove.opposite();
            validMoves.remove(oppositeMove);
            
            String stateAndMoves = String.join(",", finalState) + "\n" + validMoves;

            // Enviar estado del juego y movimientos válidos al servidor
            String response = socketPython.sendGameState(stateAndMoves);
			
			MOVE predictedMove = MOVE.valueOf(response);
			
	    	pacmanMove = predictedMove;

			/*
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
			*/
			printer.mostrarInfo("El movimiento a realizar es: " + pacmanMove.toString());
		}

		return pacmanMove;
	}    
    
	
	public String getName() {
		return "Paquita";
	}

}
