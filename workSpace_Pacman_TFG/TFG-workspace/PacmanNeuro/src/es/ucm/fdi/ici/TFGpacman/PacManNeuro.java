package es.ucm.fdi.ici.TFGpacman;


import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import pacman.controllers.PacmanController;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import pacman.game.consolePrinter.MessagePrinter;
import pacman.game.dataManager.GameStateFilter;
import pacman.game.dataManager.MovementFilter;
import pacman.game.dataManager.ReinforcementScore;

public class PacManNeuro extends PacmanController{

	
	private SocketPython socketPython;
	private GameStateFilter gameStateFilter;
	private MessagePrinter printer;
	private ReinforcementScore rScore;
	
	public PacManNeuro() {
		
		this.gameStateFilter = new GameStateFilter();
		this.printer = new MessagePrinter(true);
		
		try {
			socketPython = new SocketPython("localhost", 12345, printer);
		} catch (Exception e) {
			printer.mostrarError("Al inicializar el socket");
			System.out.println(e.getMessage());
		}
	}
	
	
    @Override
    public MOVE getMove(Game game, long timeDue) {    
    	
    	//Solo se hace la primera vez
    	if (rScore == null)
    	    rScore = new ReinforcementScore(game);
    	else
    	    rScore.update(game);
    	
    	MOVE pacmanMove = MOVE.NEUTRAL;
    	
    	if (game.isJunction(game.getPacmanCurrentNodeIndex())) {
    		//Obtener estado del juego procesado
    		List<String> filteredState = gameStateFilter.filterGameState(game.getGameState());
    		List<String> finalState = gameStateFilter.addNewVariablesToFilteredState(game, filteredState, null);
    		
    		
    		//Ponemos el score calculado por refuerzo, no el real del juego
    		finalState.set(2, String.valueOf(rScore.getScore()));
    		
    		
            // Obtener movimientos posibles sin colisiones
            MOVE[] possibleMoves = game.getPossibleMoves(game.getPacmanCurrentNodeIndex());
            List<MOVE> validMoves = new ArrayList<>(Arrays.asList(possibleMoves));
					
            // Restringir la vuelta atras
            MOVE lastMove = game.getPacmanLastMoveMade();
            MOVE oppositeMove = lastMove.opposite();
            validMoves.remove(oppositeMove);
            
            
    		//List<MOVE> validMoves = MovementFilter.getValidMoves(game);
    		
            String stateAndMoves = String.join(",", finalState) + "\n" + validMoves;

            // Enviar estado del juego y movimientos válidos al servidor
            String response = socketPython.sendAndReceivePrediction(stateAndMoves);
			
			MOVE predictedMove = MOVE.valueOf(response);
			
	    	pacmanMove = predictedMove;

			printer.printMessage("El movimiento a realizar es: " + pacmanMove.toString(), "info", 1);
		}

		return pacmanMove;
	}    
    
	
    
    @Override
    public  void postCompute() {
    	socketPython.close();
    }    
    
    
    
	public String getName() {
		return "Paquita";
	}

}
