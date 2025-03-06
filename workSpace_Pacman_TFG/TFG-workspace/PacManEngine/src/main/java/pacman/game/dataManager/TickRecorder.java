package pacman.game.dataManager;

import java.util.List;

import pacman.game.Constants.MOVE;
import pacman.game.Game;

public class TickRecorder {
	
	public TickRecorder() {
	}
	
	
	public String collectTick(Game game) {
		
		GameStateFilter gameStateFilter = new GameStateFilter(game);
		

		// Se recoge el estado del juego y se eliminan las caracteristicas que no queremos
		List<String> filteredState = gameStateFilter.filterGameState(game.getGameState());

		// Se calculan las nuevas variables que queremos en ese instante, y se añaden
		List<String> finalState = gameStateFilter.addNewVariablesToFilteredState(filteredState);

		return String.join(",", finalState);
	}
}
