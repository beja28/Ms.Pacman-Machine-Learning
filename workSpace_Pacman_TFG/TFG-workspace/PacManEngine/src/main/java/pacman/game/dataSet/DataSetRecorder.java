package pacman.game.dataSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;


public class DataSetRecorder {
    
    private List<String> totalGameStates;  //Para almacenar los distintos estados del juego
    private List<String> validGameStates;
    private List<Integer> previousScores;
    
    private final Game game;

    public DataSetRecorder(Game game) {
        this.totalGameStates = new ArrayList<>();
        this.validGameStates = new ArrayList<>();
        this.previousScores =new ArrayList<>();
        this.game = game;
    }

    
    /*
     * 	Guarda un estado del juego, si MsPacman se encuentra en un posicion, procesa los datos
     * 	elimando caracteristicas que no son utiles y agreagando/calculando nuevas caracteristicas
     * */
    public void collectGameState(MOVE pacmanMove) {
    	
    	//Se recoge el estado del juego y se eliminan las caracteristicas que no queremos
    	List<String> filteredState = filterGameState(game.getGameState());
    	
    	filteredState.add(0, pacmanMove.toString());
    	
    	//Agregar el resto de variables
    	String finalState = addNewVariablesToGameState(filteredState, previousScores);    	
    	
    	
    	
    	//Si MsPacman se encuentra en una INTERSECCION...
    	if(game.isJunction(game.getPacmanCurrentNodeIndex())) {   		
    		validGameStates.add(finalState);
    	}
    	
    	totalGameStates.add(finalState);
    	previousScores.add(game.getScore());  // Actualizar lista de puntuaciones
    }

    
    
    //Filtra un string del estado del texto, para quitar las variables que no quiero
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

        //Convertimos la lista filtrada de nuevo a un string separado por comas
        //return String.join(",", filteredState);
        return filteredState;
    }   
    
    
    
    // Calcula la diferencia de puntuación en ticks anteriores
    public int calculateScoreDifference(int currentScore, List<Integer> previousScores, int ticks) {
    	
        if (previousScores.size() >= ticks) {
            return currentScore - previousScores.get(previousScores.size() - ticks);
        }
        
        return -1;  // No se puede calcular
    }

    
    
    // Calcula la distancia del camino mas corto entre dos puntos
    public int calculateShortestPathDistance(int pacmanNode, int targetNode) {
    	//El fantasma esta en la carcel
        if (targetNode == -1) {
            return -1;
        }
        
        return game.getShortestPathDistance(pacmanNode, targetNode);
    }

    
    // Encontrar la Power Pill activa más cercana
    public int findNearestActivePP(Game game, int pacmanNode) {
        int[] activePowerPills = game.getActivePowerPillsIndices(); // Obtener todas las PP activas

        if (activePowerPills.length == 0) {
            return -1; // No hay Power Pills activas
        }

        int nearestPP = activePowerPills[0]; // Inicializar con la primera Power Pill activa
        int shortestPathDistance = game.getShortestPathDistance(pacmanNode, nearestPP); // Distancia del camino más corto

        // Recorrer las Power Pills activas para encontrar la más cercana
        for (int ppNode : activePowerPills) {
            int pathDistance = game.getShortestPathDistance(pacmanNode, ppNode);

            // Actualizar si encontramos una PP más cercana
            if (pathDistance < shortestPathDistance) {
                shortestPathDistance = pathDistance;
                nearestPP = ppNode;
            }
        }

        return nearestPP; // Retornar el nodo de la PP más cercana
    }
    
    
    
    // Calcular la distancia del camino más corto a la PP activa más cercana
    public int calculatePathDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

        if (nearestPP == -1) {
            return -1; // No hay Power Pills activas
        }

        // Calcular y devolver la distancia del camino más corto
        return game.getShortestPathDistance(pacmanNode, nearestPP);
    }

    
    // Calcular la distancia euclídea a la PP activa más cercana
    public int calculateEuclideanDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

        if (nearestPP == -1) {
            return -1; // No hay Power Pills activas
        }

        // Calcular y devolver la distancia euclídea
        return (int) game.getEuclideanDistance(pacmanNode, nearestPP);
    }
    
    
    //Calcula el numero de PowerPills restantes
    public int getRemainingPowerPills() {
    	
    	//Retorna la longitud del array de PP activas
        return game.getActivePowerPillsIndices().length;
    }

    
    

    
    // Añadir nuevas variables al estado del juego
    public String addNewVariablesToGameState(List<String> gameState, List<Integer> previousScores) {
    	//3 variables adicionales con las puntuaciones obtenidas en los 10, 25 y 50 anteriores ticks de ejecucion
    	int scoreDiff10 = calculateScoreDifference(game.getScore(), previousScores, 10);
        int scoreDiff25 = calculateScoreDifference(game.getScore(), previousScores, 25);
        int scoreDiff50 = calculateScoreDifference(game.getScore(), previousScores, 50);
        gameState.add(scoreDiff10 + "");
    	gameState.add(scoreDiff25 + "");
    	gameState.add(scoreDiff50 + "");    	

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

        return String.join(",", gameState);
    }
	
    
    
    /*
    // Recoger el estado del juego, añadir las variables nuevas y almacenarlo
    public void collectGameState(Game game) {
        String state = filterGameState(game.getGameState());
        state = addNewVariablesToGameState(game, state, previousScores);
        
        if(game.isJunction(game.getPacmanCurrentNodeIndex())) {
            validGameStates.add(state);
        }
        
        totalGameStates.add(state);
        previousScores.add(game.getScore());  // Actualizar lista de puntuaciones
    }
    */
    
    
    
    public void saveDataToCsv(String fileName, boolean show_header) throws IOException {
        List<String> fileContent = new ArrayList<>();
        String filePath = fileName + ".csv";

        // Verificar si el archivo ya existe
        boolean fileExists = Files.exists(Paths.get(filePath));

        // Si se quiere mostrar el encabezado y el archivo no existe, añadirlo
        if (show_header && !fileExists) {
            fileContent.add(String.join(",", DataSetVariables.getFinalGameState()));
        }

        // Añadir los estados del juego (filas)
        fileContent.addAll(validGameStates);

        // Guardar los datos en el archivo en modo APPEND (añadir al final)
        Files.write(Paths.get(filePath), fileContent, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
    
}