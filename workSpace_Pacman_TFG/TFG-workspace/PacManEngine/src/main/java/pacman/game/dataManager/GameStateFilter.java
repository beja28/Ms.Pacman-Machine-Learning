package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import pacman.game.Game;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;

public class GameStateFilter {

	
	public GameStateFilter() {}

    
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
    public List<String> addNewVariablesToFilteredState(Game game, List<String> gameState, LinkedList<Integer> lastJunctionScore) {

    	//Distancia del path a los fantasmas
    	for (GHOST ghost : GHOST.values()) {
    		int distanceToGhost = calculateShortestPathDistance(game, game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghost));
    		gameState.add(distanceToGhost + "");
    	}
    	
        //Ditancia euclidea y del path a la PP activa mas cercana
        //int euclideanToPpDistancia = calculateEuclideanDistanceToNearestPP(game, game.getPacmanCurrentNodeIndex());
        int pathToPpDistancia = calculateEuclideanDistanceToNearestPP(game, game.getPacmanCurrentNodeIndex());
        //gameState.add(euclideanToPpDistancia + "");
        gameState.add(pathToPpDistancia + "");
        
        //Numero de PP restantes
        int remainingPPills = getRemainingPowerPills(game);
        gameState.add(remainingPPills + "");

        int sinceLastReversal = getsinceLastReversal(game);
        gameState.add(sinceLastReversal + "");
        
        int reward = getReward(game, lastJunctionScore);
        gameState.add(reward + "");

        return gameState;
    }    
    
        
    // Calcula la distancia del "path" mas corto entre dos nodos
    public int calculateShortestPathDistance(Game game, int pacmanNode, int targetNode) {
   
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
    public int calculatePathDistanceToNearestPP(Game game, int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

        if (nearestPP == -1) {
            return -1; // No hay PP activas
        }

        // Calcula y devuelve la distancia del camino más corto
        return game.getShortestPathDistance(pacmanNode, nearestPP);
    }

    
    // Calcula la distancia euclidea a la PP activa mas cercana
    public int calculateEuclideanDistanceToNearestPP(Game game, int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP mas cercana

        if (nearestPP == -1) {
            return -1; // No hay
        }

        //Calcula y devuelve la distancia euclidea
        return (int) game.getEuclideanDistance(pacmanNode, nearestPP);
    }

    // Calcula el porcentaje de pills que ha comido
    public double getPillsEaten(Game game) {
        double remaining = (double) game.getNumberOfActivePills();
        double total = (double) game.getNumberOfPills();
        double ratio = 1.0 - (remaining / total);
        return Math.sqrt(ratio);
    }

       
    //Calcula el numero de PowerPills restantes
    public int getRemainingPowerPills(Game game) {
    	
    	//Retorna la longitud del array de PP activas
        return game.getActivePowerPillsIndices().length;
    }
    
    // Calcular hace cuanto tiempo fue la ultima inversion global
    public int getsinceLastReversal(Game game) {
    	int sinceLastReversal = game.getTotalTime() - game.getTimeOfLastGlobalReversal();
    	
    	return sinceLastReversal;
    }
    
    // Calcula la diferencia de score de una interseccion a otra. Cambiará en funcion de los rewards que se le asignen en el tramo de una interseccion a otra
    public int getReward(Game game, LinkedList<Integer> lastJunctionScore) {
    	
    	if(lastJunctionScore == null) {
    		return 0; 
    	}
    	
    	if (lastJunctionScore.isEmpty()) {
            // Casos extremos: lista nula o incompleta -> asumimos reward 0
            return 0;
        }

        if (lastJunctionScore.size() == 1) {
            return lastJunctionScore.get(0);
        }

        int scorePrev = lastJunctionScore.get(0);
        int scoreNow = lastJunctionScore.get(1);

        return scoreNow - scorePrev;
    	
    }


}
