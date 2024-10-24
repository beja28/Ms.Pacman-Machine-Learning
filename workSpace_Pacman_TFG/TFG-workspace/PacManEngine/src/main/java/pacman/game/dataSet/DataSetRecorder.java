package pacman.game.dataSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;


public class DataSetRecorder {
    
    private List<String> validGameStates;
    private List<Integer> previousScores;
    
    private final Game game;

    public DataSetRecorder(Game game) {
        this.validGameStates = new ArrayList<>();
        this.previousScores =new ArrayList<>();
        this.game = game;
    }

    
    /*
     * 	Guarda un estado del juego, si Pacman se encuentra en un posicion, procesa los datos
     * 	elimando caracteristicas que no son utiles y agreagando/calculando nuevas caracteristicas
     * */
    public void collectGameState(MOVE pacmanMove) {
    	    	
    	//Si Pacman se encuentra en una INTERSECCION...
    	if(game.isJunction(game.getPacmanCurrentNodeIndex())) {
    		
    		//Se recoge el estado del juego y se eliminan las caracteristicas que no queremos
        	List<String> filteredState = filterGameState(game.getGameState());
        	
        	//Se añade la etiqueta (la posicion de Pacman) 
        	filteredState.add(0, pacmanMove.toString());
        	
        	//Se calculan las nuevas variables que queremos, y se añaden
        	String finalState = addNewVariablesToFilteredState(filteredState);    		
    		
    		validGameStates.add(finalState);
    	}
    	
    	previousScores.add(game.getScore());
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
    
        
    // Calcula la diferencia de puntuacion en estados anteriores
    public int calculateScoreDifference(int currentScore, List<Integer> previousScores, int ticks) {
    	
        if (previousScores.size() >= ticks) {
            return currentScore - previousScores.get(previousScores.size() - ticks);
        }
                
        //En caso de que no se pueda calcular
        return -1;
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

            // Actualizar si encontramos una PP más cercana
            if (dist < shortestPathDistance) {
                shortestPathDistance = dist;
                nearestPP = ppNodo;
            }
        }

        return nearestPP;
    }
    
    
    // Calcula la distancia del camino más corto a la PP activa más cercana
    public int calculatePathDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

        if (nearestPP == -1) {
            return -1; // No hay PP activas
        }

        // Calcula y devuelve la distancia del camino más corto
        return game.getShortestPathDistance(pacmanNode, nearestPP);
    }

    
    // Calcula la distancia euclidea a la PP activa más cercana
    public int calculateEuclideanDistanceToNearestPP(int pacmanNode) {
        int nearestPP = findNearestActivePP(game, pacmanNode); // Encontrar la PP más cercana

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

    
    // Añade nuevas variables al estado del juego
    public String addNewVariablesToFilteredState(List<String> gameState) {
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
     * Funcion que añade los datos de los estados del juego calculados a un archivo .csv
     * 		- Si el archivo ya existe se añaden los estados
     * 		- Si el archivo no existe, se crea y se añaden las cabeceras de las columnas, y se añaden los estados 
     * */
    public void saveDataToCsv(String fileName, boolean show_header) throws IOException {
    	List<String> fileContent = new ArrayList<>();

        //Nombre de la carpeta donde queremos guardar los dataSets
        String folderName = "new_dataSets"; 

        //Crear ruta
        Path folderPath = Paths.get(folderName);

        //Si no existe la carpeta se crea
        if (!Files.exists(folderPath)) {
            try {
                Files.createDirectories(folderPath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Ruta completa del csv (carpeta + nombre))
        Path filePath = folderPath.resolve(fileName + ".csv");

        // Se comprueba si hay un archivo existente con ese nombre
        boolean fileExists = Files.exists(filePath);

        // Si no existe, se crea y se añade el encabezado
        if (!fileExists && show_header) {
            fileContent.add(String.join(",", DataSetVariables.getFinalGameState()));
        }

        // Se añaden los estados calculados
        fileContent.addAll(validGameStates);

        // Se uardan los datos en el archivo .csv en modo APPEND (añadir al final)
        Files.write(filePath, fileContent, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
    
}
