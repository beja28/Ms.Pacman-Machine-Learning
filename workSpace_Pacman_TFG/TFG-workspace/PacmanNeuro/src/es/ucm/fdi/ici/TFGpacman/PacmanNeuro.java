package es.ucm.fdi.ici.TFGpacman;

import java.util.HashSet;
import java.util.Set;

import pacman.controllers.PacmanController;
import pacman.game.Constants.MOVE;
import pacman.game.socket.SocketPython;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.DM;
import pacman.game.Game;

public class PacmanNeuro extends PacmanController{

	
	private SocketPython socketPython;
	
	public PacmanNeuro() {
		// Crear instancia de SocketPython
		try {
			socketPython = new SocketPython("localhost", 12345);
		} catch (Exception e) {
			System.out.println(e.getMessage());
			System.out.println("Error al inicializar el socket");
		}
	}
	
    @Override
    public MOVE getMove(Game game, long timeDue) {
     
    	//HAY QUE RECOLECTAR EL ESTADO DEL JUEGO EN TODAS LAS LLAMADAS
    	
    	if (game.isJunction(game.getPacmanCurrentNodeIndex())) {
			String filteredGameState = gameFilter.getActualGameState();
			String response = socketPython.sendGameState(filteredGameState);

			try {
				pacmanMove = MOVE.valueOf(response);
			} catch (Exception e) {
				System.out.println("Respuesta inválida del servidor: " + response);
				break;
			}
		}

		return move;

	}    
    

	
	public String getName() {
		return "Pacman Neuroanl";
	}

}
