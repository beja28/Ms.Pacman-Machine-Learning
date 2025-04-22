package pacman.game.dataManager;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import pacman.game.Game;

public abstract class BaseGameStateTracker {
	
	protected Queue<List<String>> bufferGameStates;
	
	
	public BaseGameStateTracker() {
		this.bufferGameStates = new LinkedList<>();
	}
	
	
	//Funciones que se llaman en la clase "DataManager"
	public abstract void updateGameStateTracker(Game game);	
	public abstract void storeJunctionState(Game game, List<String> pendingState);	
	public abstract String processBufferStates(Game game);
	public abstract LinkedList<Integer> getJunctionScore();
	
}