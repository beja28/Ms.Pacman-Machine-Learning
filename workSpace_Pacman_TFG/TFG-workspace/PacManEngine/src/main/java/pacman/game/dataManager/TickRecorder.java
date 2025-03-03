package pacman.game.dataManager;

import java.util.List;

import pacman.game.Constants.MOVE;
import pacman.game.Game;

public class TickRecorder {
	
	private GameStateFilter gameStateFilter;
	private Game game;
	
	public TickRecorder(Game game) {
		this.game = game;
		this.gameStateFilter = new GameStateFilter(game);
	}
	
	
	public String collectTick() {

		// Se recoge el estado del juego y se eliminan las caracteristicas que no queremos
		List<String> filteredState = gameStateFilter.filterGameState(game.getGameState());

		// Se calculan las nuevas variables que queremos en ese instante, y se añaden
		List<String> finalState = gameStateFilter.addNewVariablesToFilteredState(filteredState);

		return String.join(",", finalState);
	}
}
