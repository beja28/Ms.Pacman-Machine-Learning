package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.List;

import pacman.game.Game;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;

public class GameStateFilter {

	private Game game;
	private List<Integer> previousScores;
	private int lastJuctionIndex;
	
	public static final int MAX_TIME = 25;
	
	public GameStateFilter(Game game) {
		this.game = game;
		this.previousScores = new ArrayList<>();
		this.lastJuctionIndex = 1;
	}

    
    //Filtra un string con el estado del juego, quita las variables que no queremos y devuelve una lista con las restantes
    public List<String> filterGameState(String gameState) {
        
    	//Lista con las variables que se deben eliminar del estado del juego
        List<String> variablesAEliminar = DataSetVariables.VARIABLES_BORRAR_GAME_STATE;
        
        //Lista con todas las variables del estado del juego
        List<String> todasLasVariables = DataSetVariables.VARIABLES_GAME_STATE;
        
        //Lista para almacenar los valores filtrados
        List<String> filteredState = new ArrayList<>();
        
        //Convertimos el string del estado del juego en una lista
        String[] gameStateArray = gameState.split(",");

        //Recorremos todas las variables del estado del juego
        for (int i = 0; i < todasLasVariables.size(); i++) {
            // Solo agregamos las variables que no estan en la lista de variables a eliminar
            if (!variablesAEliminar.contains(todasLasVariables.get(i))) {
                filteredState.add(gameStateArray[i]);
            }
        }
        
        //Se devuelve en formato de lista
        return filteredState;
    }   
    
    
        
    // Añade nuevas variables al estado del juego, en ese instante de tiempo
    public List<String> addNewVariablesToFilteredState(List<String> gameState) {

    	//Distancia del path a los fantasmas
    	for (GHOST ghost : GHOST.values()) {
    		int distanceToGhost = calculateShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghost));
    		gameState.add(distanceToGhost + "");
    	}
    	
        //Ditancia euclidea y del path a la PP activa mas cercana
        int euclideanToPpDistancia = calculateEuclideanDistanceToNearestPP(game.getPacmanCurrentNodeIndex());
        int pathToPpDistancia = calculateEuclideanDistanceToNearestPP(game.getPacmanCurrentNodeIndex());
        gameState.add(euclideanToPpDistancia + "");
        gameState.add(pathToPpDistancia + "");
        
        //Numero de PP restantes
        int remainingPPills = getRemainingPowerPills();
        gameState.add(remainingPPills + "");

        return gameState;
    }

    
    //Guarda en cada tick el score correspondiente
    public void addPreviousScore(int score) {
        
    	previousScores.add(score);       
    }
    
    
    //Actualiza la posicion de la ultima interseccion
    public void updateLastJuctionIndex(int idx) {
    	
    	this.lastJuctionIndex = idx;
    }
    
    
    //Si en un estado a muerto pacman, calcular cuantos ticks anteriores han pasado hasta la ultima decision tomada
    public void updatePacmanDead() {
    	
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
    public boolean isDiffTime(List<String> state) {    	
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
    
        
    // Calcula la distancia del "path" mas corto entre dos nodos
    public int calculateShortestPathDistance(int pacmanNode, int targetNode) {
   
        if (targetNode == -1) {
            return -1;
        }
        
        return game.getShortestPathDistance(pacmanNode, targetNode);
    }

    
    // Retorma el nodo de la Power Pill activa mas cercana a Pacman
    public int findNearestActivePP(Game game, int pacmanNode) {
        int[] activePowerPills = game.getActivePowerPillsIndices(); // Obtener todas las PP activas

        if (activePowerPills.length == 0) {
            return -1; // No hay ninguna activa
        }

        int nearestPP = activePowerPills[0];	//Por defecto la primera
        int shortestPathDistance = game.getShortestPathDistance(pacmanNode, nearestPP);

        // Recorre el resto de las PP activas hasta encontrar la mas cercana
        for (int i=0; i<activePowerPills.length; i++) {
        	int ppNodo = activePowerPills[i];
            int dist = game.getShortestPathDistance(pacmanNode, ppNodo);

            // Actualizar si encontramos una PP mas cercana
            if (dist < shortestPathDistance) {
                shortestPathDistance = dist;
                nearestPP = ppNodo;
            }
        }

        return nearestPP;
    }
    
    
    // Calcula la distancia del camino mas corto a la PP activa más cercana
    public int calculatePathDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

        if (nearestPP == -1) {
            return -1; // No hay PP activas
        }

        // Calcula y devuelve la distancia del camino más corto
        return game.getShortestPathDistance(pacmanNode, nearestPP);
    }

    
    // Calcula la distancia euclidea a la PP activa mas cercana
    public int calculateEuclideanDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP mas cercana

        if (nearestPP == -1) {
            return -1; // No hay
        }

        //Calcula y devuelve la distancia euclidea
        return (int) game.getEuclideanDistance(pacmanNode, nearestPP);
    }
    
    
    //Calcula el numero de PowerPills restantes
    public int getRemainingPowerPills() {
    	
    	//Retorna la longitud del array de PP activas
        return game.getActivePowerPillsIndices().length;
    }


}
