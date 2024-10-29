package pacman.game.dataManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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
    private GameStateFilter gameStateFilter;;
    
    private final Game game;

    public DataSetRecorder(Game game) {
        this.validGameStates = new ArrayList<>();
        this.gameStateFilter = new GameStateFilter(game);
        this.game = game;
    }
    
    
    public void collectGameState(MOVE pacmanMove) {
    	
    	//Si Pacman se encuentra en una INTERSECCION...
    	if(game.isJunction(game.getPacmanCurrentNodeIndex())) {
    		
    		//Se recoge el estado del juego y se eliminan las caracteristicas que no queremos
        	List<String> filteredState = gameStateFilter.filterGameState(game.getGameState());
        	
        	//Se añade la etiqueta (la posicion de Pacman) 
        	filteredState.add(0, pacmanMove.toString());
        	
        	//Se calculan las nuevas variables que queremos, y se añaden
        	String finalState = gameStateFilter.addNewVariablesToFilteredState(filteredState);    		
    		
    		validGameStates.add(finalState);
    	}
    	
    	gameStateFilter.addPreviousScore(Integer.valueOf(game.getScore()));
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
    
    
    
    // Para contar cuentas lineas tiene un .csv
    public static int contarLineas(String fileName) {
    	
    	String folderName = "new_dataSets";
        Path folderPath = Paths.get(folderName);

        // Sino existe la carpeta se crea
        if (!Files.exists(folderPath)) {
            try {
                Files.createDirectories(folderPath);
            } catch (IOException e) {
                e.printStackTrace();
                return 0;
            }
        }

        // Ruta completa del archivo
        Path filePath = folderPath.resolve(fileName + ".csv");
        File archivo = filePath.toFile();

        // Se verifica si el archivo existe
        if (!archivo.exists()) {
            return 0;	//Si no existe, tiene 0 lineas
        }

        int lineas = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(archivo))) {
            while (br.readLine() != null) {
                lineas++;
            }
        } catch (IOException e) {
            System.out.println("Error al leer el archivo: " + e.getMessage());
            return 0;
        }

        return lineas;
    }
    
}
