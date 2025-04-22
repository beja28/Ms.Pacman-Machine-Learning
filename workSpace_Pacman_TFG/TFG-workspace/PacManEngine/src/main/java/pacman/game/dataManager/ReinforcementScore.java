package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.List;

import pacman.game.Constants;
import pacman.game.Game;

public class ReinforcementScore {	
	
	public static final int PILL_REWARD = 12;
    public static final int POWER_PILL_REWARD = 0;
    public static final int EDIBLE_GHOST_REWARD = 40;
    public static final int STEP_PENALTY = -1;
    public static final int WIN_REWARD = 50;
    public static final int PACMAN_DEATH_PENALTY = -375;
	
	private int score;
	private int activePills;
	private int actMazeIndex;
	
	private final List<Integer> scoreHistory;
    public static final int MAX_HISTORY = 50;
	
	public ReinforcementScore(Game game) {
		this.score = 0;
		this.activePills = game.getNumberOfActivePills();
		this.actMazeIndex = game.getMazeIndex();
		
		this.scoreHistory = new ArrayList<>();
        this.scoreHistory.add(score);
	}
	
	public int getScore() {
		return score;
	}
	
	public void update(Game game) {
		if (checkPill(game)) score += PILL_REWARD;
	    if (checkPowerPill(game)) score += POWER_PILL_REWARD;
	    if (checkEdibleGhost(game)) score += EDIBLE_GHOST_REWARD;
	    if (checkStep(game)) score += STEP_PENALTY;
	    if (checkWin(game)) score += WIN_REWARD;
	    //if (checkDeath(game)) score += PACMAN_DEATH_PENALTY;	//Ya se quita, cuando se restan los scores de la lista
	    
	    
	    scoreHistory.add(score);
	}
	
	
	//Comprueba si se ha comido una pill
	private boolean checkPill(Game game) {
		int activePillsAct = game.getNumberOfActivePills();
		
	    if (activePillsAct < this.activePills) {
	    	this.activePills = activePillsAct;
	        return true;
	    }
	    
	    this.activePills = activePillsAct;
	    return false;
	}

	
	//Comprueba si se ha comido una powerPill
	private boolean checkPowerPill(Game game) {
	    if (game.wasPowerPillEaten()) {
	        return true;
	    }
	    return false;
	}

	private boolean checkEdibleGhost(Game game) {
		for (Constants.GHOST g : Constants.GHOST.values()) {
			if (game.wasGhostEaten(g)) {
		        return true;
		    }
        }
	    
	    return false;
	}

	//Siempre devuelve true porque con cada tick se mueve
	private boolean checkStep(Game game) {
	    return true;
	}

	//Detecta si cambia de tablero
	private boolean checkWin(Game game) {
		int actMaze = game.getMazeIndex();
	    if (actMaze != this.actMazeIndex) {
	    	this.actMazeIndex = actMaze;
	        return true;
	    }
	    return false;
	}

	private boolean checkDeath(Game game) {
	    if (game.wasPacManEaten()) {
	        return true;
	    }
	    return false;
	}
	
	
	//Resta score a los ultimos scores de la lista
	public void pacmanEatenPenalizePreviousScores(int ticks) {
        int size = scoreHistory.size();
        int start = Math.max(size - ticks, 0);

        for (int i = start; i < size; i++) {
            scoreHistory.set(i, scoreHistory.get(i) - PACMAN_DEATH_PENALTY);
        }
    }
	
	
	//Resta score a los ultimos scores de la lista
	public void ghostEatenIncreasePreviousScores(int ticks) {
        int size = scoreHistory.size();
        int start = Math.max(size - ticks, 0);

        for (int i = start; i < size; i++) {
            scoreHistory.set(i, scoreHistory.get(i) + EDIBLE_GHOST_REWARD);
        }
    }

	
	//Calcula la diferencia de puntuacion entre dos momentos distintos
    public int calculateScoreDifference(int baseTick, int targetTick) {
        if (baseTick < 0 || targetTick < 0 || baseTick >= scoreHistory.size() || targetTick >= scoreHistory.size()) return 0;

        return scoreHistory.get(targetTick) - scoreHistory.get(baseTick);
    }

    public int getLastScore() {
        if (scoreHistory.isEmpty()) return 0;
        return scoreHistory.get(scoreHistory.size() - 1);
    }

    public int getScoreAt(int tickOffset) {
        int index = scoreHistory.size() - tickOffset - 1;
        if (index < 0) return 0;
        return scoreHistory.get(index);
    }

    public int getHistorySize() {
        return scoreHistory.size();
    }

    public boolean hasEnoughHistory(int requiredTicks) {
        return scoreHistory.size() >= requiredTicks;
    }

}
