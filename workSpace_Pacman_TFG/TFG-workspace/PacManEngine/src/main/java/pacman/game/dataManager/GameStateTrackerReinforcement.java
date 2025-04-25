package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;
import pacman.game.Constants;
import pacman.game.Game;
import pacman.game.Constants.DM;

public class GameStateTrackerReinforcement extends BaseGameStateTracker{
	
	private int lastJuctionIndex;
    private LinkedList<Integer> lastJunctionsTimes = new LinkedList<>();
	public static final int MAX_TIME = 25;
	private ReinforcementScore reinforScore;
	private final List<GhostEaten> ghostsEaten = new ArrayList<>();
	private final List<Boolean> pacmanEaten;
	
	public GameStateTrackerReinforcement() {
		super();
		
		this.lastJuctionIndex = 1;
		this.reinforScore = null;
		this.pacmanEaten = new ArrayList<>();
	}
		
	@Override
	public void updateGameStateTracker(Game game) {		
		
		if (reinforScore == null)
			reinforScore = new ReinforcementScore(game);
	    else
	    	reinforScore.update(game);
		
		updatePacmanDead(game);
		updateGhostsDead(game);
	}

	
	//Se llama cuando es una interseccion
	@Override
	public void storeJunctionState(Game game, List<String> pendingState) {
		int stateTick = Integer.parseInt(pendingState.get(1));
		updateLastJuctionIndex(game.getPacmanCurrentNodeIndex(), stateTick);	//Actualizar ultima interseccion
		this.bufferGameStates.add(pendingState);	//Anyadir estado a la cola
	}
	

	@Override
	public String processBufferStates(Game game) {
		String st = null;
		
		//Como mucho solo se puede procesar un estado en un instante
        if (!bufferGameStates.isEmpty()) {
        	
        	//Se obtiene el primer elemento de la cola sin eliminarlo
            List<String> state = bufferGameStates.peek();
            
            //Se comprueba si el estado ha pasado sufiente tiempo
            if (isDiffTime(game, state)) {
                //SE PONE EL SCORE POR REFUERZO
                int stateTick = Integer.parseInt(state.get(1));		//Momento en el que se guardo el estado
                int tickOffset = game.getTotalTime() - stateTick;	//Hace cuanto se guardo
                
                int refScore = reinforScore.getScoreAt(tickOffset);
                state.set(2, String.valueOf(refScore));
                
                // en vez de mirar ultimos 25 ticks, hay que mirar entre intersecciones
                if (ghostsEaten.size() >= 26) {
                    int fromTick = lastJunctionsTimes.getFirst();
                    int toTick = lastJunctionsTimes.getLast();

                    GhostEaten eatenInRange = checkGhostsEatenBuffer(fromTick, toTick);

                    boolean g1 = eatenInRange.isGhost1Eaten();
                    boolean g2 = eatenInRange.isGhost2Eaten();
                    boolean g3 = eatenInRange.isGhost3Eaten();
                    boolean g4 = eatenInRange.isGhost4Eaten();

                    state.add(String.valueOf(g1));
                    state.add(String.valueOf(g2));
                    state.add(String.valueOf(g3));
                    state.add(String.valueOf(g4));
                }
                
                st = String.join(",", state);
                bufferGameStates.poll();	//Se elimina de la cola
            }
        }
        
        return st;
	}
    
    public void addJunctionTime(Integer time) {
        if (lastJunctionsTimes.size() >= 2) {
            lastJunctionsTimes.removeFirst(); // Elimina el más antiguo
        }
        lastJunctionsTimes.addLast(time); // Añade el nuevo al final
    }
    
    //Actualiza la posicion de la ultima interseccion
    public void updateLastJuctionIndex(int idx, int tick) {
    	addJunctionTime(tick);
    	this.lastJuctionIndex = idx;
    }
    
    
    //Si en un estado a muerto pacman, calcula cuantos ticks anteriores han pasado hasta la ultima decision tomada
    public void updatePacmanDead(Game game) {
    	
    	if(game.wasPacManEaten()) {
    		int actIdx = game.getPacmanCurrentNodeIndex();
    		double dist = game.getDistance(lastJuctionIndex, actIdx, DM.PATH);
    		
    		reinforScore.pacmanEatenPenalizePreviousScores((int) dist);
    	}    	
    }
    
    
    public void updateGhostsDead(Game game) {
    	
    	GameStateTrackerReinforcement.GhostEaten eaten = new GameStateTrackerReinforcement.GhostEaten(false, false, false, false);
    	
    	if (game.wasGhostEaten(Constants.GHOST.BLINKY)) {
    		eaten.setGhost1Eaten(true);
	    }
    	if (game.wasGhostEaten(Constants.GHOST.PINKY)) {
    		eaten.setGhost2Eaten(true);
	    }
    	if (game.wasGhostEaten(Constants.GHOST.INKY)) {
    		eaten.setGhost3Eaten(true);
	    }
    	if (game.wasGhostEaten(Constants.GHOST.SUE)) {
    		eaten.setGhost4Eaten(true);
	    }
    	
    	if(eaten.anyGhostEaten()) {
    		int actIdx = game.getPacmanCurrentNodeIndex();
    		double dist = game.getDistance(lastJuctionIndex, actIdx, DM.PATH);
    		
    		reinforScore.ghostEatenIncreasePreviousScores((int) dist);
    	}
    	
    	this.ghostsEaten.add(eaten);
    }
        
    
    public GhostEaten checkGhostsEatenBuffer(int fromTick, int toTick) {
        boolean ghost1 = false;
        boolean ghost2 = false;
        boolean ghost3 = false;
        boolean ghost4 = false;

        for (int i = fromTick; i <= toTick; i++) {
            GhostEaten snapshot = ghostsEaten.get(i);
            ghost1 |= snapshot.isGhost1Eaten();
            ghost2 |= snapshot.isGhost2Eaten();
            ghost3 |= snapshot.isGhost3Eaten();
            ghost4 |= snapshot.isGhost4Eaten();
        }

        return new GhostEaten(ghost1, ghost2, ghost3, ghost4);
    }
 
    
    //Comprueba si un estado lleva suficiente tiempo guardado en la cola
    public boolean isDiffTime(Game game, List<String> state) {
    	
    	String strTime = state.get(1); //Representa la columna TotalTime
    	int time = Integer.parseInt(strTime);
    	
    	return (game.getTotalTime() - time >= MAX_TIME);
    }
    
    
    public boolean isValidGameState(List<String> state) {
    	/*
    	if (!reinforScore.hasEnoughHistory(MAX_TIME)) return false;

        int baseScore = reinforScore.getScoreAt(MAX_TIME);
        int scoreDiff10 = reinforScore.getScoreAt(10) - baseScore;
        int scoreDiff25 = reinforScore.getScoreAt(0) - baseScore;

        return (0.35 * scoreDiff10 + 0.65 * scoreDiff25) >= 0;
        */
    	return true;
    }
    
    
    public static class GhostEaten {
        private boolean ghost1Eaten;
        private boolean ghost2Eaten;
        private boolean ghost3Eaten;
        private boolean ghost4Eaten;

        public GhostEaten(boolean ghost1Eaten, boolean ghost2Eaten, boolean ghost3Eaten, boolean ghost4Eaten) {
            this.ghost1Eaten = ghost1Eaten;
            this.ghost2Eaten = ghost2Eaten;
            this.ghost3Eaten = ghost3Eaten;
            this.ghost4Eaten = ghost4Eaten;
        }

        public boolean isGhost1Eaten() {
            return ghost1Eaten;
        }

        public boolean isGhost2Eaten() {
            return ghost2Eaten;
        }

        public boolean isGhost3Eaten() {
            return ghost3Eaten;
        }

        public boolean isGhost4Eaten() {
            return ghost4Eaten;
        }
        
        public void setGhost1Eaten(boolean b) {
            this.ghost1Eaten = b;
        }

        public void setGhost2Eaten(boolean b) {
            this.ghost2Eaten = b;
        }

        public void setGhost3Eaten(boolean b) {
            this.ghost3Eaten = b;
        }

        public void setGhost4Eaten(boolean b) {
            this.ghost4Eaten = b;
        }

        @Override
        public String toString() {
            return "GhostEaten{" +
                   "ghost1Eaten=" + ghost1Eaten +
                   ", ghost2Eaten=" + ghost2Eaten +
                   ", ghost3Eaten=" + ghost3Eaten +
                   ", ghost4Eaten=" + ghost4Eaten +
                   '}';
        }
        
        public boolean anyGhostEaten() {
            return ghost1Eaten || ghost2Eaten || ghost3Eaten || ghost4Eaten;
        }
    }
	
}
