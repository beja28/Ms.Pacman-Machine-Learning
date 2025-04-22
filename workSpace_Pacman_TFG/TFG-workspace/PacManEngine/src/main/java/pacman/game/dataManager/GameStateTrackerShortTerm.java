package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import pacman.game.Game;
import pacman.game.Constants.DM;

public class GameStateTrackerShortTerm extends BaseGameStateTracker{
	
	private List<Integer> previousScores;
	private int lastJuctionIndex;
	public static final int MAX_TIME = 25;
	
	public GameStateTrackerShortTerm() {
		super();
		
		this.previousScores = new ArrayList<>();
		this.lastJuctionIndex = 1;
	}
	
	
	//Actualiza las variables privadas de esta implementacion
	@Override
	public void updateGameStateTracker(Game game) {
		addPreviousScore(game.getScore());
		updatePacmanDead(game);
	}

	
	//Se llama cuando es una interseccion
	@Override
	public void storeJunctionState(Game game, List<String> pendingState) {
		
		updateLastJuctionIndex(game.getPacmanCurrentNodeIndex());	//Actualizar ultima interseccion
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
            	
            	//Se comprueba si es valido
            	if(isValidGameState(state)) {	//Se guarda
            		st = String.join(",", state);
            	}
                bufferGameStates.poll();	//Se elimina de la cola
            }
        }
        
        return st;
	}
	
	
	
	public void addPreviousScore(int score) {
        
    	previousScores.add(score);       
    }
    
    
    //Actualiza la posicion de la ultima interseccion
    public void updateLastJuctionIndex(int idx) {
    	
    	this.lastJuctionIndex = idx;
    }
    
    
    //Si en un estado a muerto pacman, calcular cuantos ticks anteriores han pasado hasta la ultima decision tomada
    public void updatePacmanDead(Game game) {
    	
    	if(game.wasPacManEaten()) {
    		int actIdx = game.getPacmanCurrentNodeIndex();
    		double dist = game.getDistance(lastJuctionIndex, actIdx, DM.PATH);
    		//Se resta la puntuacion a esos estados anteriores
        	//decrementScoreIfEaten(dist, 1000);
        	game.decreaseTotalScore(100);
    	}    	
    }
    

    
    public void decrementScoreIfEaten(double dist, int decrement) {
    	
        int size = previousScores.size();
        int elementsToDecrement = Math.min((int) dist, size);

        for (int i = size - elementsToDecrement; i < size; i++) {
            previousScores.set(i, previousScores.get(i) - decrement);
        }
    }

    
    
    //Comprueba si un estado lleva suficiente tiempo guardado en la cola
    public boolean isDiffTime(Game game, List<String> state) {    	
    	boolean diffTime = false;
    	
    	String strTime = state.get(1); //Representa la columna TotalTime
    	int time = Integer.parseInt(strTime);
    	
    	if(game.getTotalTime() - time >= MAX_TIME) {
    		diffTime = true;
    	}
    	
		
		return diffTime;
    }
    
    
    public boolean isValidGameState(List<String> state) {
        boolean valid = false;

        // Score inicial
        int initialScore = previousScores.get(previousScores.size() - MAX_TIME);

        // Diferencia de score en distintos momentos
        int scoreDiff10 = calculateScoreDifference(initialScore, 15);
        int scoreDiff25 = calculateScoreDifference(initialScore, 0);
        
        /*
        if(scoreDiff10 < 0 || scoreDiff25 < 0) {
        	System.out.println("Se han jugado" + (game.getTotalTime()));
        	System.out.println("Aqui le han comio");
        }
        */
        
        valid = (0.35*scoreDiff10 + 0.65*scoreDiff25) >= 0;

        return valid;
    }
    
    // Muestra la diferencia de score que hay desde que se realizo el movimiento hasta el tick (100 - ticks)
    public int calculateScoreDifference(int initialScore, int ticks) {  
        return previousScores.get(previousScores.size() - ticks -1) - initialScore;
    }



	@Override
	public LinkedList<Integer> getJunctionScore() {
		// TODO Auto-generated method stub
		return null;
	}
	
	

}
