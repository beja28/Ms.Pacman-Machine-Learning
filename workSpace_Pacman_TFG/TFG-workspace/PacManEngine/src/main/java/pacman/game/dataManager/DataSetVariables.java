package pacman.game.dataManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataSetVariables {

	//Variables que retorna el estado del juego
	public static final List<String> VARIABLES_GAME_STATE = Arrays.asList(
	        "mazeIndex", "totalTime", "score", "currentLevelTime", "levelCount",
	        "pacmanCurrentNodeIndex", "pacmanLastMoveMade", "pacmanLivesRemaining",
	        "pacmanReceivedExtraLife", "ghost1NodeIndex", "ghost1EdibleTime", "ghost1LairTime",
	        "ghost1LastMove", "ghost2NodeIndex", "ghost2EdibleTime", "ghost2LairTime",
	        "ghost2LastMove", "ghost3NodeIndex", "ghost3EdibleTime", "ghost3LairTime",
	        "ghost3LastMove", "ghost4NodeIndex", "ghost4EdibleTime", "ghost4LairTime",
	        "ghost4LastMove", "pillsState", "powerPillsState", "timeOfLastGlobalReversal",
	        "pacmanWasEaten", "ghost1Eaten", "ghost2Eaten", "ghost3Eaten", "ghost4Eaten",
	        "pillWasEaten", "powerPillWasEaten"
	);

	
	//Variables que quiero quitar del estado del juego
	public static final List<String> VARIABLES_BORRAR_GAME_STATE = Arrays.asList(
	        "currentLevelTime", "levelCount", "pillsState", "pillWasEaten", 
	        "powerPillWasEaten", "powerPillsState", "pacmanLivesRemaining", 
	        "pacmanReceivedExtraLife", "mazeIndex"
	);

	
	// Variables que quiero añadir al estado del juego
	public static final List<String> VARIABLES_AGREGAR_GAME_STATE = Arrays.asList(
	        "ghost1Distance", "ghost2Distance", "ghost3Distance", "ghost4Distance", "euclideanDistanceToPp",
	        "pathDistanceToPp", "remainingPp"
	);
	
	
	//Reibe dos listas de strings, y borra los strings de la segunda lista en la primera lista
	public static List<String> restarListas(List<String> stringsTotales, List<String> stringsBorrar) {
	    // Creamos una nueva lista basada en variablesGameState
	    List<String> listaFiltrada = new ArrayList<>(stringsTotales);

	    // Restamos eliminando las variables que están en variablesBorrarGameState
	    listaFiltrada.removeAll(stringsBorrar);

	    // Retornamos la lista resultante
	    return listaFiltrada;
	}
	
	
	//Recibe dos listas de strings y agrega los strings de la segunda lista a la primera
	public static List<String> agregarListas(List<String> stringsTotales, List<String> stringsAgregar) {
	    // Creamos una nueva lista
	    List<String> listaConAñadidos = new ArrayList<>(stringsTotales);

	    // Añadimos las variables
	    listaConAñadidos.addAll(stringsAgregar);

	    // Se retorna la lista resultante
	    return listaConAñadidos;
	}
	
	
	public static List<String> getFinalGameState(){
		//Elimnar encabezados del GameState que no queremos
		List<String> filterGameState = restarListas(VARIABLES_GAME_STATE, VARIABLES_BORRAR_GAME_STATE);
		
		//Agregar los nuevos encabezados
		List<String> finalGameState = agregarListas(filterGameState, VARIABLES_AGREGAR_GAME_STATE);
		
		//Agregar la etiqueta
		finalGameState.add(0, "PacmanMove");
		
		return finalGameState;
	}
	
}
